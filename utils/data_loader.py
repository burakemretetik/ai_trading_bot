# utils/data_loader.py - Data acquisition
import pandas as pd
import ccxt
from config import API_KEY, API_SECRET
import logging

logger = logging.getLogger(__name__)

class DataLoader:
    """Class for loading market data."""
    
    def __init__(self, symbol, timeframe):
        """Initialize DataLoader with trading pair and timeframe."""
        self.symbol = symbol
        self.timeframe = timeframe
        self.exchange = self._initialize_exchange()
    
    def _initialize_exchange(self):
        """Initialize exchange connection."""
        # Using Binance as an example, but can be changed to other exchanges
        exchange = ccxt.binance({
            'apiKey': API_KEY,
            'secret': API_SECRET,
            'enableRateLimit': True,
        })
        return exchange
    
    def get_historical_data(self, limit=500):
        """Fetch historical OHLCV data."""
        try:
            logger.info(f"Fetching historical data for {self.symbol}, timeframe: {self.timeframe}")
            ohlcv = self.exchange.fetch_ohlcv(self.symbol, self.timeframe, limit=limit)
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            logger.info(f"Successfully fetched {len(df)} candlesticks")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            return pd.DataFrame()