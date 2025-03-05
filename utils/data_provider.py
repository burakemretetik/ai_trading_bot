# utils/data_provider.py - Abstract data provider interface
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
import pandas as pd
import logging
from datetime import datetime
import os
import ccxt
import yfinance as yf

from config import API_KEY, API_SECRET

logger = logging.getLogger(__name__)

class DataProvider(ABC):
    """Abstract base class for data providers."""
    
    def __init__(self, symbol: str, timeframe: str):
        """Initialize the data provider.
        
        Args:
            symbol: Trading symbol/pair
            timeframe: Candlestick timeframe
        """
        self.symbol = symbol
        self.timeframe = timeframe
    
    @abstractmethod
    def get_historical_data(self, 
                          limit: int = 1000, 
                          start_date: Optional[str] = None, 
                          end_date: Optional[str] = None) -> pd.DataFrame:
        """Fetch historical OHLCV data.
        
        Args:
            limit: Maximum number of candles to fetch
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            
        Returns:
            DataFrame with columns ['open', 'high', 'low', 'close', 'volume'] and DatetimeIndex
        """
        pass
    
    @abstractmethod
    def get_current_price(self) -> float:
        """Get current market price.
        
        Returns:
            Current price as a float
        """
        pass
    
    def format_and_validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure data is properly formatted and handle missing values.
        
        Args:
            df: Raw DataFrame with OHLCV data
            
        Returns:
            Cleaned and validated DataFrame
        """
        if df.empty:
            logger.warning("Empty DataFrame received")
            return df
        
        # Ensure columns exist and are properly named
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in df.columns:
                logger.error(f"Required column '{col}' missing from data")
                return pd.DataFrame()
        
        # Check for NaN values
        for col in required_columns:
            nan_count = df[col].isna().sum()
            if nan_count > 0:
                logger.warning(f"Found {nan_count} NaN values in {col} column, filling with forward fill")
                df[col] = df[col].fillna(method='ffill')
                
                # If the first rows had NaN values, backward fill
                if df[col].isna().any():
                    df[col] = df[col].fillna(method='bfill')
        
        return df
    
    def save_data_to_csv(self, df: pd.DataFrame, filename: Optional[str] = None) -> str:
        """Save data to CSV file.
        
        Args:
            df: DataFrame to save
            filename: Optional filename, defaults to symbol_timeframe_date.csv
            
        Returns:
            Path to saved file
        """
        if df.empty:
            logger.warning("Cannot save empty DataFrame")
            return ""
        
        # Create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)
        
        # Generate filename if not provided
        if not filename:
            # Replace problematic characters in symbol
            clean_symbol = self.symbol.replace('/', '_').replace('-', '_')
            timestamp = datetime.now().strftime('%Y%m%d')
            filename = f"data/{clean_symbol}_{self.timeframe}_{timestamp}.csv"
        
        # Save to CSV
        df.to_csv(filename)
        logger.info(f"Data saved to {filename}")
        
        return filename


class CCXTDataProvider(DataProvider):
    """Data provider using CCXT for cryptocurrency exchanges."""
    
    def __init__(self, symbol: str, timeframe: str, exchange_id: str = 'binance'):
        """Initialize CCXT data provider.
        
        Args:
            symbol: Trading symbol/pair (e.g., 'BTC/USDT')
            timeframe: Candlestick timeframe (e.g., '1h', '1d')
            exchange_id: CCXT exchange ID (default: 'binance')
        """
        super().__init__(symbol, timeframe)
        
        # Format symbol if needed (e.g., BTCUSDT -> BTC/USDT)
        self.original_symbol = symbol
        self.symbol = self._format_symbol(symbol)
        
        # Initialize exchange
        self.exchange_id = exchange_id
        self.exchange = self._initialize_exchange()
    
    def _format_symbol(self, symbol: str) -> str:
        """Convert concatenated symbol to CCXT format with slash."""
        if '/' in symbol:
            return symbol
        
        # Most common format is BTCUSDT -> BTC/USDT
        if len(symbol) >= 6 and 'USDT' in symbol:
            base = symbol.replace('USDT', '')
            return f"{base}/USDT"
        
        # Try to find other quote currencies (BTC, ETH, etc.)
        for quote in ['BTC', 'ETH', 'BNB', 'USDC']:
            if symbol.endswith(quote):
                base = symbol.replace(quote, '')
                return f"{base}/{quote}"
        
        # Default: just return the original
        return symbol
    
    def _initialize_exchange(self) -> ccxt.Exchange:
        """Initialize exchange connection."""
        try:
            # Initialize exchange with API keys if available
            exchange_class = getattr(ccxt, self.exchange_id)
            exchange = exchange_class({
                'apiKey': API_KEY,
                'secret': API_SECRET,
                'enableRateLimit': True,
                'timeout': 30000,  # 30 seconds
            })
            
            # Test connection
            exchange.fetchStatus()
            logger.info(f"Successfully connected to {exchange.name}")
            
            return exchange
        except Exception as e:
            logger.error(f"Error initializing exchange: {e}")
            # Return a basic exchange instance without API keys for public endpoints
            exchange_class = getattr(ccxt, self.exchange_id)
            return exchange_class({'enableRateLimit': True})
    
    def get_historical_data(self, 
                          limit: int = 1000, 
                          start_date: Optional[str] = None, 
                          end_date: Optional[str] = None) -> pd.DataFrame:
        """Fetch historical OHLCV data using CCXT.
        
        Args:
            limit: Maximum number of candles to fetch
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            logger.info(f"Fetching historical data for {self.symbol} (original: {self.original_symbol}), timeframe: {self.timeframe}")
            
            # Convert start_date to timestamp if provided
            since = None
            if start_date:
                # Convert to timestamp in milliseconds
                since = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
                logger.info(f"Using start date: {start_date} ({since})")
            
            # Load specific market
            try:
                self.exchange.load_markets(False)
                market = self.exchange.market(self.symbol)
                logger.info(f"Successfully loaded market for {self.symbol}")
            except Exception as e:
                logger.error(f"Error loading market for {self.symbol}: {e}")
                # Try searching for similar symbols as a fallback
                all_symbols = self.exchange.symbols
                if all_symbols:
                    base_currency = self.symbol.split('/')[0] if '/' in self.symbol else self.original_symbol[:3]
                    similar_symbols = [s for s in all_symbols if base_currency in s]
                    if similar_symbols:
                        logger.info(f"Could not find exact symbol. Similar symbols: {similar_symbols[:5]}...")
                return pd.DataFrame()
            
            # Fetch OHLCV data
            ohlcv = self.exchange.fetch_ohlcv(self.symbol, self.timeframe, since=since, limit=limit)
            
            # Check if we got any data
            if not ohlcv or len(ohlcv) == 0:
                logger.error(f"No data returned from exchange for {self.symbol}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Apply date filters if specified
            if start_date:
                start_date = pd.to_datetime(start_date)
                df = df[df.index >= start_date]
            
            if end_date:
                end_date = pd.to_datetime(end_date)
                df = df[df.index <= end_date]
            
            logger.info(f"Successfully fetched {len(df)} candlesticks from {df.index.min()} to {df.index.max()}")
            
            # Check if we have enough data
            if len(df) < 100:
                logger.warning(f"Retrieved only {len(df)} candlesticks, which may not be enough for reliable analysis")
            
            # Validate and format data
            return self.format_and_validate_data(df)
            
        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            return pd.DataFrame()
    
    def get_current_price(self) -> float:
        """Get current market price using CCXT.
        
        Returns:
            Current price as a float
        """
        try:
            ticker = self.exchange.fetch_ticker(self.symbol)
            price = ticker['last']
            logger.info(f"Current price for {self.symbol}: {price}")
            return price
        except Exception as e:
            logger.error(f"Error fetching current price: {e}")
            return 0.0


class YFinanceDataProvider(DataProvider):
    """Data provider using Yahoo Finance."""
    
    def __init__(self, symbol: str, timeframe: str):
        """Initialize Yahoo Finance data provider.
        
        Args:
            symbol: Trading symbol (e.g., 'BTC-USD')
            timeframe: Candlestick timeframe (e.g., '1h', '1d')
        """
        super().__init__(symbol, timeframe)
        
        # Map timeframe to yfinance interval format
        self.interval_map = {
            '1m': '1m',
            '5m': '5m', 
            '15m': '15m',
            '30m': '30m',
            '1h': '1h',
            '4h': '4h',
            '1d': '1d',
            '1w': '1wk',
            '1mo': '1mo'
        }
        
        # Check if timeframe is supported
        if timeframe not in self.interval_map:
            logger.warning(f"Timeframe {timeframe} not directly supported by yfinance. Using default '1h'")
            self.yf_interval = '1h'
        else:
            self.yf_interval = self.interval_map[timeframe]
    
    def get_historical_data(self, 
                          limit: int = 1000, 
                          start_date: Optional[str] = None, 
                          end_date: Optional[str] = None) -> pd.DataFrame:
        """Fetch historical OHLCV data using Yahoo Finance.
        
        Args:
            limit: Maximum number of candles to fetch (used to calculate start date if not provided)
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            logger.info(f"Fetching historical data for {self.symbol} using Yahoo Finance, interval: {self.yf_interval}")
            
            # Set default dates if not provided
            if not end_date:
                end_date = datetime.now().strftime('%Y-%m-%d')
                
            if not start_date:
                # Calculate start date based on limit and interval
                if self.yf_interval == '1d':
                    days_back = limit
                elif self.yf_interval == '1h':
                    days_back = limit // 24 + 1  # Add 1 for safety
                elif self.yf_interval in ['1m', '5m', '15m', '30m']:
                    days_back = 7  # Yahoo only provides 7 days of intraday data
                else:
                    days_back = 365  # Default to 1 year
                
                # Calculate start date
                start_datetime = datetime.now() - pd.Timedelta(days=days_back)
                start_date = start_datetime.strftime('%Y-%m-%d')
            
            logger.info(f"Date range: {start_date} to {end_date}")
            
            # Download data from Yahoo Finance
            data = yf.download(
                self.symbol,
                start=start_date,
                end=end_date,
                interval=self.yf_interval,
                progress=False
            )
            
            if data.empty:
                logger.error(f"No data returned from Yahoo Finance for {self.symbol}")
                return pd.DataFrame()
            
            # Process data to match our format
            # Yahoo returns OHLCV data with columns: Open, High, Low, Close, Adj Close, Volume
            data = data.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
            
            # Drop 'Adj Close' column
            if 'Adj Close' in data.columns:
                data = data.drop('Adj Close', axis=1)
            
            logger.info(f"Successfully fetched {len(data)} candlesticks from {data.index.min()} to {data.index.max()}")
            
            # Validate and format data
            return self.format_and_validate_data(data)
            
        except Exception as e:
            logger.error(f"Error fetching historical data from Yahoo Finance: {e}")
            return pd.DataFrame()
    
    def get_current_price(self) -> float:
        """Get current market price using Yahoo Finance.
        
        Returns:
            Current price as a float
        """
        try:
            ticker = yf.Ticker(self.symbol)
            # Get the most recent price
            current = ticker.history(period='1d')
            if current.empty:
                return 0.0
                
            price = current['Close'].iloc[-1]
            logger.info(f"Current price for {self.symbol}: {price}")
            return price
        except Exception as e:
            logger.error(f"Error fetching current price from Yahoo Finance: {e}")
            return 0.0


class SimulatedDataProvider(DataProvider):
    """Data provider generating simulated price data."""
    
    def __init__(self, symbol: str = 'SIM/USD', timeframe: str = '1h', volatility: float = 0.01, 
               drift: float = 0.0001, start_price: float = 1000.0):
        """Initialize simulated data provider.
        
        Args:
            symbol: Simulated symbol name
            timeframe: Candlestick timeframe (e.g., '1h', '1d')
            volatility: Price volatility parameter
            drift: Price drift parameter (daily)
            start_price: Initial price
        """
        super().__init__(symbol, timeframe)
        self.volatility = volatility
        self.drift = drift
        self.start_price = start_price
        self.current_price = start_price
    
    def get_historical_data(self, 
                          limit: int = 1000, 
                          start_date: Optional[str] = None, 
                          end_date: Optional[str] = None) -> pd.DataFrame:
        """Generate simulated historical OHLCV data.
        
        Args:
            limit: Number of candles to generate
            start_date: Start date in 'YYYY-MM-DD' format (overrides limit)
            end_date: End date in 'YYYY-MM-DD' format (default: today)
            
        Returns:
            DataFrame with simulated OHLCV data
        """
        import numpy as np
        
        logger.info(f"Generating simulated data for {self.symbol}, timeframe: {self.timeframe}")
        
        # Generate dates
        if end_date:
            end_date = pd.to_datetime(end_date)
        else:
            end_date = datetime.now()
            
        if start_date:
            start_date = pd.to_datetime(start_date)
            # Generate dates from start_date to end_date
            freq = self._get_frequency_string()
            dates = pd.date_range(start=start_date, end=end_date, freq=freq)
        else:
            # Generate 'limit' number of dates back from end_date
            freq = self._get_frequency_string()
            if self.timeframe == '1d':
                # For daily data, we need to go back 'limit' days
                start_date = end_date - pd.Timedelta(days=limit)
            elif self.timeframe == '1h':
                # For hourly data, we need to go back 'limit' hours
                start_date = end_date - pd.Timedelta(hours=limit)
            elif self.timeframe == '1m':
                # For minute data, we need to go back 'limit' minutes
                start_date = end_date - pd.Timedelta(minutes=limit)
            else:
                # Default fallback
                start_date = end_date - pd.Timedelta(days=limit)
                
            dates = pd.date_range(start=start_date, end=end_date, freq=freq)
        
        # Generate price series (random walk with drift)
        np.random.seed(42)  # For reproducibility
        
        # Adjust parameters based on timeframe
        timeframe_multiplier = self._get_timeframe_multiplier()
        adjusted_drift = self.drift * timeframe_multiplier
        adjusted_volatility = self.volatility * np.sqrt(timeframe_multiplier)
        
        # Generate returns
        returns = np.random.normal(adjusted_drift, adjusted_volatility, len(dates))
        
        # Add some patterns for ML to detect
        t = np.arange(len(dates))
        cycles = 0.02 * np.sin(t/24*2*np.pi) + 0.01 * np.sin(t/168*2*np.pi)
        returns = returns + cycles
        
        # Generate price
        price = self.start_price * np.cumprod(1 + returns)
        
        # Create OHLCV dataframe
        df = pd.DataFrame(index=dates)
        df['close'] = price
        
        # Generate open, high, low prices
        df['open'] = np.roll(df['close'], 1)
        df.loc[df.index[0], 'open'] = self.start_price  # First open price
        
        # Generate high and low prices
        price_range = adjusted_volatility * price
        df['high'] = df[['open', 'close']].max(axis=1) + price_range * np.random.random(len(df))
        df['low'] = df[['open', 'close']].min(axis=1) - price_range * np.random.random(len(df))
        
        # Ensure high >= open/close and low <= open/close
        df['high'] = df[['high', 'open', 'close']].max(axis=1)
        df['low'] = df[['low', 'open', 'close']].min(axis=1)
        
        # Generate volume (correlated with absolute returns)
        base_volume = 1000000  # Base volume
        abs_returns = np.abs(returns)
        volume_factor = 1 + 5 * abs_returns  # Higher volume on bigger price moves
        df['volume'] = base_volume * volume_factor * np.random.lognormal(0, 0.5, len(df))
        
        logger.info(f"Generated {len(df)} simulated candlesticks from {df.index[0]} to {df.index[-1]}")
        
        # Update current price
        self.current_price = df['close'].iloc[-1]
        
        return df
    
    def _get_frequency_string(self) -> str:
        """Get pandas frequency string based on timeframe."""
        if self.timeframe == '1m':
            return 'T'  # Minute
        elif self.timeframe == '5m':
            return '5T'
        elif self.timeframe == '15m':
            return '15T'
        elif self.timeframe == '30m':
            return '30T'
        elif self.timeframe == '1h':
            return 'H'  # Hour
        elif self.timeframe == '4h':
            return '4H'
        elif self.timeframe == '1d':
            return 'D'  # Day
        elif self.timeframe == '1w':
            return 'W'  # Week
        elif self.timeframe == '1mo':
            return 'M'  # Month
        else:
            logger.warning(f"Unknown timeframe: {self.timeframe}, defaulting to hourly")
            return 'H'
    
    def _get_timeframe_multiplier(self) -> float:
        """Get multiplier for adjusting parameters based on timeframe.
        
        Returns:
            Multiplier relative to 1 day
        """
        if self.timeframe == '1m':
            return 1/1440  # 1/minutes in day
        elif self.timeframe == '5m':
            return 5/1440
        elif self.timeframe == '15m':
            return 15/1440
        elif self.timeframe == '30m':
            return 30/1440
        elif self.timeframe == '1h':
            return 1/24
        elif self.timeframe == '4h':
            return 4/24
        elif self.timeframe == '1d':
            return 1
        elif self.timeframe == '1w':
            return 7
        elif self.timeframe == '1mo':
            return 30
        else:
            return 1/24  # Default to hourly
    
    def get_current_price(self) -> float:
        """Get current simulated market price.
        
        Returns:
            Current price as a float
        """
        return self.current_price


# Factory function to create the appropriate data provider
def create_data_provider(data_source: str, symbol: str, timeframe: str, **kwargs) -> DataProvider:
    """Factory function to create a data provider.
    
    Args:
        data_source: Type of data provider ('ccxt', 'yfinance', 'simulated')
        symbol: Trading symbol/pair
        timeframe: Candlestick timeframe
        **kwargs: Additional parameters for specific data providers
        
    Returns:
        DataProvider instance
    """
    if data_source.lower() == 'ccxt':
        exchange_id = kwargs.get('exchange_id', 'binance')
        return CCXTDataProvider(symbol, timeframe, exchange_id)
    elif data_source.lower() == 'yfinance':
        return YFinanceDataProvider(symbol, timeframe)
    elif data_source.lower() == 'simulated':
        volatility = kwargs.get('volatility', 0.01)
        drift = kwargs.get('drift', 0.0001)
        start_price = kwargs.get('start_price', 1000.0)
        return SimulatedDataProvider(symbol, timeframe, volatility, drift, start_price)
    else:
        logger.error(f"Unknown data source: {data_source}")
        # Default to simulated data
        return SimulatedDataProvider(symbol, timeframe)