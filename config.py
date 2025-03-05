# config.py - Configuration settings
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API credentials
API_KEY = os.getenv("API_KEY", "")
API_SECRET = os.getenv("API_SECRET", "")

# Trading parameters
SYMBOL = "BTCUSDT"  # Example: Bitcoin/USDT pair
TIMEFRAME = "1h"    # Example: 1-hour candlesticks
RISK_PER_TRADE = 0.02  # 2% risk per trade

# Backtesting parameters
INITIAL_CAPITAL = 10000  # Starting capital for backtesting
COMMISSION = 0.001       # 0.1% commission fee

# Note: Removed circular import. The DataLoader will be imported in files that need it.