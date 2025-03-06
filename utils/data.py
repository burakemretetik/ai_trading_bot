# utils/data.py - Unified data provider for all data sources
import pandas as pd
import numpy as np
import logging
import os
from datetime import datetime
from typing import Optional, Dict, Any, List

# Optional imports based on data source
try:
    import ccxt
except ImportError:
    ccxt = None

try:
    import yfinance as yf
except ImportError:
    yf = None

# Setup logging
logger = logging.getLogger(__name__)

class DataProvider:
    """Unified data provider for multiple data sources."""
    
    def __init__(self, symbol: str, timeframe: str, data_source: str = 'ccxt', **kwargs):
        """
        Initialize data provider.
        
        Args:
            symbol: Trading symbol/pair (e.g., 'BTC/USDT' or 'BTCUSDT')
            timeframe: Candlestick timeframe (e.g., '1h', '1d')
            data_source: Data source type ('ccxt', 'yfinance', 'simulated', 'csv')
            **kwargs: Additional parameters for specific data sources
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.data_source = data_source.lower()
        self.params = kwargs
        
        # Format symbol for CCXT if needed
        self.original_symbol = symbol
        if self.data_source == 'ccxt':
            self.symbol = self._format_symbol(symbol)
            
            # Initialize exchange connection
            self.exchange_id = kwargs.get('exchange_id', 'binance')
            self.api_key = kwargs.get('api_key', '')
            self.api_secret = kwargs.get('api_secret', '')
            self.exchange = self._initialize_exchange()
        
        # Set simulated data parameters
        elif self.data_source == 'simulated':
            self.volatility = kwargs.get('volatility', 0.01)
            self.drift = kwargs.get('drift', 0.0001)
            self.start_price = kwargs.get('start_price', 1000.0)
            self.current_price = self.start_price
    
    def _format_symbol(self, symbol: str) -> str:
        """Convert concatenated symbol to CCXT format with slash."""
        if '/' in symbol:
            return symbol
        
        # Most common format is BTCUSDT -> BTC/USDT
        if len(symbol) >= 6 and 'USDT' in symbol:
            base = symbol.replace('USDT', '')
            return f"{base}/USDT"
        
        # Try to find other quote currencies
        for quote in ['BTC', 'ETH', 'BNB', 'USDC']:
            if symbol.endswith(quote):
                base = symbol.replace(quote, '')
                return f"{base}/{quote}"
        
        return symbol
    
    def _initialize_exchange(self):
        """Initialize exchange connection for CCXT."""
        if ccxt is None:
            logger.error("CCXT library not installed. Cannot use CCXT data source.")
            return None
            
        try:
            exchange_class = getattr(ccxt, self.exchange_id)
            exchange = exchange_class({
                'apiKey': self.api_key,
                'secret': self.api_secret,
                'enableRateLimit': True,
                'timeout': 30000,
            })
            
            # Test connection
            exchange.fetchStatus()
            logger.info(f"Successfully connected to {exchange.name}")
            return exchange
        except Exception as e:
            logger.error(f"Error initializing exchange: {e}")
            # Return a basic exchange instance without API keys
            exchange_class = getattr(ccxt, self.exchange_id)
            return exchange_class({'enableRateLimit': True})
    
    def get_historical_data(self, 
                          limit: int = 1000, 
                          start_date: Optional[str] = None, 
                          end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Get historical OHLCV data from the specified data source.
        
        Args:
            limit: Maximum number of candles to fetch
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            
        Returns:
            DataFrame with OHLCV data
        """
        if self.data_source == 'ccxt':
            return self._get_ccxt_data(limit, start_date, end_date)
        elif self.data_source == 'yfinance':
            return self._get_yfinance_data(limit, start_date, end_date)
        elif self.data_source == 'simulated':
            return self._get_simulated_data(limit, start_date, end_date)
        elif self.data_source == 'csv':
            return self._get_csv_data(start_date, end_date)
        else:
            logger.error(f"Unsupported data source: {self.data_source}")
            return pd.DataFrame()
    
    def _get_ccxt_data(self, limit: int, start_date: Optional[str], end_date: Optional[str]) -> pd.DataFrame:
        """Fetch historical data from CCXT."""
        if ccxt is None or self.exchange is None:
            logger.error("CCXT not available")
            return pd.DataFrame()
            
        try:
            logger.info(f"Fetching CCXT data for {self.symbol}, timeframe: {self.timeframe}")
            
            # Convert start_date to timestamp if provided
            since = None
            if start_date:
                since = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
            
            # Fetch OHLCV data
            ohlcv = self.exchange.fetch_ohlcv(self.symbol, self.timeframe, since=since, limit=limit)
            
            if not ohlcv or len(ohlcv) == 0:
                logger.error(f"No data returned from exchange for {self.symbol}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Apply date filters
            if start_date:
                start_date = pd.to_datetime(start_date)
                df = df[df.index >= start_date]
            
            if end_date:
                end_date = pd.to_datetime(end_date)
                df = df[df.index <= end_date]
                
            return self._clean_data(df)
            
        except Exception as e:
            logger.error(f"Error fetching CCXT data: {e}")
            return pd.DataFrame()
    
    def _get_yfinance_data(self, limit: int, start_date: Optional[str], end_date: Optional[str]) -> pd.DataFrame:
        """Fetch historical data from Yahoo Finance."""
        if yf is None:
            logger.error("yfinance not available")
            return pd.DataFrame()
            
        try:
            logger.info(f"Fetching Yahoo Finance data for {self.symbol}, interval: {self.timeframe}")
            
            # Map timeframe to yfinance format
            interval_map = {
                '1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m',
                '1h': '1h', '4h': '4h', '1d': '1d', '1w': '1wk', '1mo': '1mo'
            }
            yf_interval = interval_map.get(self.timeframe, '1d')
            
            # Set default dates if not provided
            if not end_date:
                end_date = datetime.now().strftime('%Y-%m-%d')
                
            if not start_date:
                # Calculate start date based on limit
                days_back = limit
                if self.timeframe == '1h':
                    days_back = limit // 24 + 1
                
                start_datetime = datetime.now() - pd.Timedelta(days=days_back)
                start_date = start_datetime.strftime('%Y-%m-%d')
            
            # Download data
            data = yf.download(
                self.symbol,
                start=start_date,
                end=end_date,
                interval=yf_interval,
                progress=False
            )
            
            if data.empty:
                logger.error(f"No data returned from Yahoo Finance for {self.symbol}")
                return pd.DataFrame()
            
            # Process data
            data = data.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
            
            if 'Adj Close' in data.columns:
                data = data.drop('Adj Close', axis=1)
                
            return self._clean_data(data)
            
        except Exception as e:
            logger.error(f"Error fetching Yahoo Finance data: {e}")
            return pd.DataFrame()
    
    def _get_simulated_data(self, limit: int, start_date: Optional[str], end_date: Optional[str]) -> pd.DataFrame:
        """Generate simulated price data."""
        try:
            logger.info(f"Generating simulated data, timeframe: {self.timeframe}")
            
            # Generate dates
            if end_date:
                end_date = pd.to_datetime(end_date)
            else:
                end_date = datetime.now()
                
            if start_date:
                start_date = pd.to_datetime(start_date)
            else:
                # Calculate start date based on limit and timeframe
                if self.timeframe == '1d':
                    start_date = end_date - pd.Timedelta(days=limit)
                elif self.timeframe == '1h':
                    start_date = end_date - pd.Timedelta(hours=limit)
                elif self.timeframe == '1m':
                    start_date = end_date - pd.Timedelta(minutes=limit)
                else:
                    start_date = end_date - pd.Timedelta(days=limit)
            
            # Map timeframe to pandas frequency
            freq_map = {
                '1m': '1min', '5m': '5min', '15m': '15min', '30m': '30min',
                '1h': '1H', '4h': '4H', '1d': '1D', '1w': '1W', '1mo': '1M'
            }
            freq = freq_map.get(self.timeframe, '1D')
            
            # Generate dates
            dates = pd.date_range(start=start_date, end=end_date, freq=freq)
            
            # Generate price data
            np.random.seed(42)  # For reproducibility
            
            # Calculate parameters based on timeframe
            if self.timeframe == '1d':
                tf_multiplier = 1
            elif self.timeframe == '1h':
                tf_multiplier = 1/24
            elif self.timeframe == '1m':
                tf_multiplier = 1/1440
            else:
                tf_multiplier = 1
                
            # Generate returns
            drift = self.drift * tf_multiplier
            volatility = self.volatility * np.sqrt(tf_multiplier)
            returns = np.random.normal(drift, volatility, len(dates))
            
            # Add some patterns
            t = np.arange(len(dates))
            cycle = 0.01 * np.sin(t/20 * 2 * np.pi)
            returns = returns + cycle
            
            # Generate price
            price = self.start_price * np.cumprod(1 + returns)
            
            # Create OHLCV dataframe
            df = pd.DataFrame(index=dates)
            df['close'] = price
            
            # Generate open, high, low, volume
            df['open'] = df['close'].shift(1)
            df.loc[df.index[0], 'open'] = self.start_price
            
            price_range = volatility * price
            df['high'] = df[['open', 'close']].max(axis=1) + price_range * np.random.random(len(df))
            df['low'] = df[['open', 'close']].min(axis=1) - price_range * np.random.random(len(df))
            df['high'] = df[['high', 'open', 'close']].max(axis=1)
            df['low'] = df[['low', 'open', 'close']].min(axis=1)
            
            # Generate volume
            base_volume = 1000000
            abs_returns = np.abs(returns)
            volume_factor = 1 + 5 * abs_returns
            df['volume'] = base_volume * volume_factor * np.random.lognormal(0, 0.5, len(df))
            
            # Update current price
            self.current_price = df['close'].iloc[-1]
            
            return self._clean_data(df)
            
        except Exception as e:
            logger.error(f"Error generating simulated data: {e}")
            return pd.DataFrame()
    
    def _get_csv_data(self, start_date: Optional[str], end_date: Optional[str]) -> pd.DataFrame:
        """Load data from CSV file."""
        try:
            filepath = self.params.get('filepath', f'data/{self.symbol.replace("/", "_")}.csv')
            
            if not os.path.exists(filepath):
                logger.error(f"CSV file not found: {filepath}")
                return pd.DataFrame()
                
            logger.info(f"Loading data from CSV: {filepath}")
            
            # Load data
            df = pd.read_csv(filepath)
            
            # Check if timestamp/date column exists
            date_col = None
            for col in ['timestamp', 'date', 'time', 'datetime']:
                if col in df.columns:
                    date_col = col
                    break
            
            if date_col is None:
                logger.error("No date/timestamp column found in CSV")
                return pd.DataFrame()
                
            # Convert to datetime and set as index
            df[date_col] = pd.to_datetime(df[date_col])
            df.set_index(date_col, inplace=True)
            
            # Apply date filters
            if start_date:
                start_date = pd.to_datetime(start_date)
                df = df[df.index >= start_date]
            
            if end_date:
                end_date = pd.to_datetime(end_date)
                df = df[df.index <= end_date]
                
            # Ensure OHLCV columns exist
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in required_columns:
                if col not in df.columns:
                    # Try to find column with case-insensitive match
                    for df_col in df.columns:
                        if df_col.lower() == col:
                            df[col] = df[df_col]
                            break
            
            return self._clean_data(df[['open', 'high', 'low', 'close', 'volume']])
            
        except Exception as e:
            logger.error(f"Error loading CSV data: {e}")
            return pd.DataFrame()
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate data."""
        if df.empty:
            return df
            
        # Check required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in df.columns:
                logger.error(f"Required column '{col}' missing from data")
                return pd.DataFrame()
        
        # Check for and handle NaN values
        for col in required_columns:
            if df[col].isna().any():
                logger.warning(f"Found NaN values in {col} column, filling")
                df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
        
        return df
    
    def get_current_price(self) -> float:
        """Get current market price."""
        if self.data_source == 'ccxt':
            try:
                ticker = self.exchange.fetch_ticker(self.symbol)
                return ticker['last']
            except:
                return 0.0
        elif self.data_source == 'yfinance':
            try:
                ticker = yf.Ticker(self.symbol)
                current = ticker.history(period='1d')
                return current['Close'].iloc[-1]
            except:
                return 0.0
        elif self.data_source == 'simulated':
            return self.current_price
        else:
            return 0.0
    
    def save_to_csv(self, df: pd.DataFrame, filepath: Optional[str] = None) -> str:
        """Save data to CSV file."""
        if df.empty:
            logger.warning("Cannot save empty DataFrame")
            return ""
        
        # Create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)
        
        # Generate filename if not provided
        if not filepath:
            symbol_str = self.symbol.replace('/', '_')
            timestamp = datetime.now().strftime('%Y%m%d')
            filepath = f"data/{symbol_str}_{self.timeframe}_{timestamp}.csv"
        
        # Save to CSV
        df.to_csv(filepath)
        logger.info(f"Data saved to {filepath}")
        
        return filepath

# Factory function to get a data provider with less configuration
def get_data_provider(symbol, timeframe, data_source='ccxt', **kwargs):
    """Create a data provider with the specified configuration."""
    return DataProvider(symbol, timeframe, data_source, **kwargs)