# utils/data.py - Improved data provider with better error handling
import pandas as pd
import numpy as np
import logging
import os
from datetime import datetime
from typing import Optional, Dict, Any, List, Union, Tuple

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
        """
        Convert symbol to CCXT format with slash.
        
        Improved to handle more symbol formats and provide better logging.
        """
        if '/' in symbol:
            return symbol
        
        # Common quote currencies
        quote_currencies = ['USDT', 'USD', 'BTC', 'ETH', 'BNB', 'USDC', 'BUSD']
        
        # Try to identify quote currency
        for quote in quote_currencies:
            if symbol.endswith(quote):
                base = symbol[:-len(quote)]
                if base:  # Ensure we have a base currency
                    return f"{base}/{quote}"
        
        # If we can't determine, log a warning and return the original
        logger.warning(f"Could not determine base/quote for symbol: {symbol}. Using as-is.")
        return symbol
    
    def _initialize_exchange(self):
        """
        Initialize exchange connection for CCXT.
        
        Improved error handling with better fallbacks.
        """
        if ccxt is None:
            logger.error("CCXT library not installed. Cannot use CCXT data source.")
            return None
            
        try:
            # Check if exchange exists
            if not hasattr(ccxt, self.exchange_id):
                logger.error(f"Exchange '{self.exchange_id}' not found in CCXT")
                return None
                
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
            
            # Try a basic instance if API keys failed
            try:
                if hasattr(ccxt, self.exchange_id):
                    exchange_class = getattr(ccxt, self.exchange_id)
                    return exchange_class({'enableRateLimit': True})
                else:
                    logger.error(f"Exchange '{self.exchange_id}' not found in CCXT")
                    return None
            except Exception as e2:
                logger.error(f"Failed to create basic exchange instance: {e2}")
                return None
    
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
        """Fetch historical data from CCXT with improved error handling."""
        if ccxt is None or self.exchange is None:
            logger.error("CCXT not available or exchange not initialized")
            return pd.DataFrame()
            
        try:
            logger.info(f"Fetching CCXT data for {self.symbol}, timeframe: {self.timeframe}")
            
            # Validate symbol exists on the exchange
            markets = self.exchange.load_markets()
            if self.symbol not in markets:
                logger.error(f"Symbol {self.symbol} not found on exchange {self.exchange.name}")
                return pd.DataFrame()
            
            # Convert start_date to timestamp if provided
            since = None
            if start_date:
                try:
                    since = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
                except ValueError:
                    logger.error(f"Invalid start_date format: {start_date}, expected 'YYYY-MM-DD'")
                    return pd.DataFrame()
            
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
                try:
                    start_date = pd.to_datetime(start_date)
                    df = df[df.index >= start_date]
                except ValueError:
                    logger.warning(f"Could not parse start_date: {start_date}")
            
            if end_date:
                try:
                    end_date = pd.to_datetime(end_date)
                    df = df[df.index <= end_date]
                except ValueError:
                    logger.warning(f"Could not parse end_date: {end_date}")
                
            return self._clean_data(df)
            
        except Exception as e:
            logger.error(f"Error fetching CCXT data: {e}")
            return pd.DataFrame()
    
    def _get_yfinance_data(self, limit: int, start_date: Optional[str], end_date: Optional[str]) -> pd.DataFrame:
        """Fetch historical data from Yahoo Finance with improved validations."""
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
            yf_interval = interval_map.get(self.timeframe)
            
            if yf_interval is None:
                logger.error(f"Unsupported timeframe for Yahoo Finance: {self.timeframe}")
                return pd.DataFrame()
            
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
            
            # Validate symbol exists
            try:
                ticker = yf.Ticker(self.symbol)
                # Get a small amount of data to check if symbol exists
                test_data = ticker.history(period="1d")
                if test_data.empty:
                    logger.error(f"Symbol {self.symbol} not found on Yahoo Finance")
                    return pd.DataFrame()
            except Exception as e:
                logger.error(f"Error validating Yahoo Finance symbol: {e}")
                return pd.DataFrame()
            
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
        """Generate simulated price data with improved random seed handling."""
        try:
            logger.info(f"Generating simulated data, timeframe: {self.timeframe}")
            
            # Generate dates
            if end_date:
                try:
                    end_date = pd.to_datetime(end_date)
                except ValueError:
                    logger.warning(f"Invalid end_date format: {end_date}, using current time")
                    end_date = datetime.now()
            else:
                end_date = datetime.now()
                
            if start_date:
                try:
                    start_date = pd.to_datetime(start_date)
                except ValueError:
                    logger.warning(f"Invalid start_date format: {start_date}, calculating from limit")
                    start_date = None
            
            if start_date is None:
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
                '1h': '1h', '4h': '4h', '1d': '1D', '1w': '1W', '1mo': '1M'
            }
            freq = freq_map.get(self.timeframe, '1D')
                        
            # Generate dates
            dates = pd.date_range(start=start_date, end=end_date, freq=freq)
            
            # Use a unique seed based on symbol and timeframe for reproducibility
            seed = hash(f"{self.symbol}_{self.timeframe}") % (2**32)
            np.random.seed(seed)
            
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
        """Load data from CSV file with improved column handling."""
        try:
            filepath = self.params.get('filepath', f'data/{self.symbol.replace("/", "_")}.csv')
            
            if not os.path.exists(filepath):
                logger.error(f"CSV file not found: {filepath}")
                return pd.DataFrame()
                
            logger.info(f"Loading data from CSV: {filepath}")
            
            # Load data
            df = pd.read_csv(filepath)
            
            if df.empty:
                logger.error(f"CSV file is empty: {filepath}")
                return pd.DataFrame()
            
            # Check if timestamp/date column exists
            date_col = None
            date_col_candidates = ['timestamp', 'date', 'time', 'datetime', 'Date', 'Timestamp', 'DateTime']
            
            for col in date_col_candidates:
                if col in df.columns:
                    date_col = col
                    break
            
            if date_col is None:
                logger.error("No date/timestamp column found in CSV")
                return pd.DataFrame()
                
            # Convert to datetime and set as index
            try:
                df[date_col] = pd.to_datetime(df[date_col])
                df.set_index(date_col, inplace=True)
            except Exception as e:
                logger.error(f"Error converting date column: {e}")
                return pd.DataFrame()
            
            # Apply date filters
            if start_date:
                try:
                    start_date = pd.to_datetime(start_date)
                    df = df[df.index >= start_date]
                except ValueError:
                    logger.warning(f"Could not parse start_date: {start_date}")
            
            if end_date:
                try:
                    end_date = pd.to_datetime(end_date)
                    df = df[df.index <= end_date]
                except ValueError:
                    logger.warning(f"Could not parse end_date: {end_date}")
                
            # Map column names to standardized names
            column_mapping = {}
            
            # Define possible column names for each required field
            required_columns = {
                'open': ['open', 'Open', 'OPEN', 'opening', 'Opening'],
                'high': ['high', 'High', 'HIGH', 'highest', 'Highest'],
                'low': ['low', 'Low', 'LOW', 'lowest', 'Lowest'],
                'close': ['close', 'Close', 'CLOSE', 'closing', 'Closing'],
                'volume': ['volume', 'Volume', 'VOLUME', 'vol', 'Vol']
            }
            
            # Create column mapping
            for std_name, possible_names in required_columns.items():
                # First check exact matches
                for name in possible_names:
                    if name in df.columns:
                        column_mapping[name] = std_name
                        break
            
            # Rename columns if mapping was created
            if column_mapping:
                df = df.rename(columns=column_mapping)
            
            # Check if all required columns exist after renaming
            missing_columns = [col for col in ['open', 'high', 'low', 'close', 'volume'] if col not in df.columns]
            if missing_columns:
                logger.error(f"Missing required columns in CSV: {missing_columns}")
                return pd.DataFrame()
                
            return self._clean_data(df[['open', 'high', 'low', 'close', 'volume']])
            
        except Exception as e:
            logger.error(f"Error loading CSV data: {e}")
            return pd.DataFrame()
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate data with better handling of invalid values."""
        if df.empty:
            return df
            
        # Check required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Required columns missing from data: {missing_columns}")
            return pd.DataFrame()
        
        # Check for and handle NaN values
        for col in required_columns:
            if df[col].isna().any():
                na_count = df[col].isna().sum()
                logger.warning(f"Found {na_count} NaN values in {col} column, filling")
                df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
        
        # Check for and handle negative or zero values
        for col in ['open', 'high', 'low', 'close']:
            if (df[col] <= 0).any():
                zero_count = (df[col] <= 0).sum()
                logger.warning(f"Found {zero_count} negative/zero values in {col} column, replacing with min positive value")
                min_positive = df[df[col] > 0][col].min()
                df.loc[df[col] <= 0, col] = min_positive
        
        # Ensure high >= low
        invalid_hl = (df['high'] < df['low']).sum()
        if invalid_hl > 0:
            logger.warning(f"Found {invalid_hl} cases where high < low, swapping values")
            mask = df['high'] < df['low']
            df.loc[mask, ['high', 'low']] = df.loc[mask, ['low', 'high']].values
        
        # Ensure high >= open/close
        df['high'] = df[['high', 'open', 'close']].max(axis=1)
        
        # Ensure low <= open/close
        df['low'] = df[['low', 'open', 'close']].min(axis=1)
        
        # Ensure volume is non-negative
        if (df['volume'] < 0).any():
            logger.warning("Found negative volume values, converting to absolute values")
            df['volume'] = df['volume'].abs()
        
        return df
    
    def get_current_price(self) -> float:
        """Get current market price with improved error handling."""
        try:
            if self.data_source == 'ccxt':
                if self.exchange is None:
                    logger.error("Exchange not initialized")
                    return 0.0
                    
                ticker = self.exchange.fetch_ticker(self.symbol)
                return ticker['last']
            elif self.data_source == 'yfinance':
                if yf is None:
                    logger.error("yfinance not available")
                    return 0.0
                    
                ticker = yf.Ticker(self.symbol)
                current = ticker.history(period='1d')
                if current.empty:
                    logger.error(f"Could not retrieve current price for {self.symbol}")
                    return 0.0
                return current['Close'].iloc[-1]
            elif self.data_source == 'simulated':
                return self.current_price
            else:
                logger.error(f"Cannot get current price from data source: {self.data_source}")
                return 0.0
        except Exception as e:
            logger.error(f"Error getting current price: {e}")
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