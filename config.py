# trading_bot/
# ├── config.py            # Configuration settings
# ├── data/                # Data storage
# ├── main.py              # Entry point
# ├── models/              # ML models
# ├── strategies/          # Trading strategies
# ├── utils/               # Utility functions
# └── requirements.txt     # Dependencies

# config.py - Configuration settings
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API credentials
API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")

# Trading parameters
SYMBOL = "BTCUSDT"  # Example: Bitcoin/USDT pair
TIMEFRAME = "1h"    # Example: 1-hour candlesticks
RISK_PER_TRADE = 0.02  # 2% risk per trade

# Backtesting parameters
INITIAL_CAPITAL = 10000  # Starting capital for backtesting
COMMISSION = 0.001       # 0.1% commission fee

# main.py - Entry point
from config import *
from utils.data_loader import DataLoader
from strategies.simple_moving_average import SMAStrategy
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading_bot.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def main():
    """Main function to run the trading bot."""
    logger.info("Starting trading bot...")
    
    # Initialize data loader
    data_loader = DataLoader(symbol=SYMBOL, timeframe=TIMEFRAME)
    
    # Load historical data
    historical_data = data_loader.get_historical_data()
    
    # Initialize strategy
    strategy = SMAStrategy(short_window=20, long_window=50)
    
    # Generate trading signals
    signals = strategy.generate_signals(historical_data)
    
    # Execute trades based on signals
    # TODO: Implement trade execution
    
    logger.info("Trading bot finished.")

if __name__ == "__main__":
    main()

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

# strategies/simple_moving_average.py - Strategy implementation
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class SMAStrategy:
    """Simple Moving Average crossover strategy."""
    
    def __init__(self, short_window, long_window):
        """Initialize strategy with window sizes."""
        self.short_window = short_window
        self.long_window = long_window
    
    def generate_signals(self, data):
        """Generate buy/sell signals based on SMA crossover."""
        logger.info(f"Generating signals using SMA strategy: short={self.short_window}, long={self.long_window}")
        
        # Make a copy of the data
        df = data.copy()
        
        # Calculate short and long moving averages
        df['short_ma'] = df['close'].rolling(window=self.short_window, min_periods=1).mean()
        df['long_ma'] = df['close'].rolling(window=self.long_window, min_periods=1).mean()
        
        # Initialize signals to 0
        df['signal'] = 0
        
        # Generate signals: 1 for buy, -1 for sell
        df.loc[df['short_ma'] > df['long_ma'], 'signal'] = 1
        df.loc[df['short_ma'] < df['long_ma'], 'signal'] = -1
        
        # Generate crossover signals
        df['position'] = df['signal'].diff()
        
        # Drop NaN values
        df.dropna(inplace=True)
        
        logger.info(f"Generated {len(df[df['position'] == 2])} buy signals and {len(df[df['position'] == -2])} sell signals")
        
        return df

# requirements.txt - Dependencies
"""
ccxt==3.0.0
pandas==2.0.0
numpy==1.23.0
python-dotenv==1.0.0
scikit-learn==1.2.0
matplotlib==3.7.0
seaborn==0.12.0
jupyter==1.0.0
pytest==7.3.0
"""