# utils/data_loader.py - Data acquisition with proper CCXT symbol format
import pandas as pd
import ccxt
from config import API_KEY, API_SECRET
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class DataLoader:
    """Class for loading market data."""
    
    def __init__(self, symbol, timeframe):
        """Initialize DataLoader with trading pair and timeframe."""
        # Convert symbol to CCXT format if needed
        self.original_symbol = symbol
        self.symbol = self._format_symbol(symbol)
        self.timeframe = timeframe
        self.exchange = self._initialize_exchange()
    
    def _format_symbol(self, symbol):
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
    
    def _initialize_exchange(self):
        """Initialize exchange connection."""
        try:
            # Using Binance as an example, but can be changed to other exchanges
            exchange = ccxt.binance({
                'apiKey': API_KEY,
                'secret': API_SECRET,
                'enableRateLimit': True,
                'timeout': 30000,  # 30 seconds
            })
            
            # Test connection - but don't load all markets yet
            exchange.fetchStatus()
            logger.info(f"Successfully connected to {exchange.name}")
            
            return exchange
        except Exception as e:
            logger.error(f"Error initializing exchange: {e}")
            # Return a basic exchange instance without API keys for public endpoints
            return ccxt.binance({'enableRateLimit': True})
    
    def get_historical_data(self, limit=1000, start_date=None, end_date=None):
        """Fetch historical OHLCV data."""
        try:
            logger.info(f"Fetching historical data for {self.symbol} (original: {self.original_symbol}), timeframe: {self.timeframe}")
            
            # If start_date is provided, convert to timestamp
            since = None
            if start_date:
                # Convert to timestamp in milliseconds
                since = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
                logger.info(f"Using start date: {start_date} ({since})")
            
            # Load specific market instead of all markets
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
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            return pd.DataFrame()