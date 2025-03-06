# strategies/sma_strategy.py - Simple Moving Average strategy
import pandas as pd
import numpy as np
import logging
from strategies.strategy import Strategy

logger = logging.getLogger(__name__)

class SMAStrategy(Strategy):
    """Simple Moving Average crossover strategy."""
    
    def __init__(self, short_window=20, long_window=50):
        """
        Initialize SMA strategy with window sizes.
        
        Args:
            short_window: Short moving average window size
            long_window: Long moving average window size
        """
        super().__init__("SMA")
        self.short_window = short_window
        self.long_window = long_window
    
    def generate_signals(self, data):
        """
        Generate buy/sell signals based on SMA crossover.
        
        Args:
            data: DataFrame with price data
            
        Returns:
            DataFrame with signals
        """
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
        
        # Generate position changes (this will be used by the backtester)
        df['position'] = df['signal'].diff()
        
        # Fill NaN values
        df['position'] = df['position'].fillna(0)
        
        # Count signals
        buy_signals = (df['position'] == 2).sum()
        sell_signals = (df['position'] == -2).sum()
        
        logger.info(f"Generated {buy_signals} buy signals and {sell_signals} sell signals")
        
        return df