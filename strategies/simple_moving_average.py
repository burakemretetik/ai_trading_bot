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