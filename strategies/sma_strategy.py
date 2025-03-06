# strategies/sma_strategy.py - Fixed Simple Moving Average strategy
import pandas as pd
import numpy as np
import logging
from typing import Optional, Dict, Any
from strategies.strategy import Strategy

logger = logging.getLogger(__name__)

class SMAStrategy(Strategy):
    """Simple Moving Average crossover strategy with improved validation."""
    
    def __init__(self, short_window=20, long_window=50, filter_noise=True, confirm_days=1):
        """
        Initialize SMA strategy with window sizes.
        
        Args:
            short_window: Short moving average window size
            long_window: Long moving average window size
            filter_noise: Whether to filter out noise with additional constraints
            confirm_days: Number of days to confirm a trend before generating signal
        """
        super().__init__("SMA")
        
        # Validate window sizes
        if short_window >= long_window:
            logger.warning(f"Short window ({short_window}) should be less than long window ({long_window}). Swapping values.")
            short_window, long_window = long_window, short_window
        
        self.short_window = short_window
        self.long_window = long_window
        self.filter_noise = filter_noise
        self.confirm_days = max(1, confirm_days)  # Ensure at least 1 day
    
    def _validate_input_data(self, data: pd.DataFrame) -> bool:
        """
        Validate input data for the strategy.
        
        Args:
            data: DataFrame to validate
            
        Returns:
            Boolean indicating if data is valid
        """
        # Check if DataFrame is empty
        if data.empty:
            logger.error("Input data is empty")
            return False
        
        # Check required columns
        if 'close' not in data.columns:
            logger.error("Missing required 'close' column")
            return False
        
        # Check for sufficient data points based on long window
        min_required = self.long_window + self.confirm_days + 5
        if len(data) < min_required:
            logger.error(f"Insufficient data points: {len(data)}. Need at least {min_required}.")
            return False
        
        # Check for NaN values in close
        if data['close'].isna().any():
            logger.warning("Close prices contain NaN values. These will be filled.")
        
        return True
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate buy/sell signals based on SMA crossover.
        
        Args:
            data: DataFrame with price data
            
        Returns:
            DataFrame with signals
        """
        logger.info(f"Generating signals using SMA strategy: short={self.short_window}, long={self.long_window}")
        
        # Validate input data
        if not self._validate_input_data(data):
            # Return original dataframe with empty signals
            df = data.copy()
            df['signal'] = 0
            df['position'] = 0
            logger.error("Cannot generate signals due to invalid input data")
            return df
        
        # Make a copy of the data
        df = data.copy()
        
        # Fill missing values in close price if any (using updated methods)
        df['close'] = df['close'].ffill().bfill()
        
        # Calculate short and long moving averages with proper minimum periods
        df['short_ma'] = df['close'].rolling(window=self.short_window, min_periods=self.short_window).mean()
        df['long_ma'] = df['close'].rolling(window=self.long_window, min_periods=self.long_window).mean()
        
        # Fill NaN values in moving averages to avoid comparison issues
        df['short_ma'] = df['short_ma'].ffill()
        df['long_ma'] = df['long_ma'].ffill()
        
        # Initialize signals to 0
        df['signal'] = 0
        df['sma_cross'] = 0
        
        # Make sure we have enough data for the moving averages
        start_idx = max(self.short_window, self.long_window)
        if start_idx >= len(df):
            logger.warning("Not enough data for moving averages")
            df['position'] = 0
            return df
            
        # Generate crossover points - avoid direct Series comparison
        # Instead, compare values row by row
        for i in range(start_idx, len(df)):
            if df['short_ma'].iloc[i] > df['long_ma'].iloc[i]:
                df.loc[df.index[i], 'sma_cross'] = 1
            elif df['short_ma'].iloc[i] < df['long_ma'].iloc[i]:
                df.loc[df.index[i], 'sma_cross'] = -1
        
        # Generate signals only on changes (crossovers)
        df['cross_change'] = df['sma_cross'].diff()
        
        # Apply trend confirmation if required
        if self.confirm_days > 1:
            # We need to confirm the cross for multiple days
            for i in range(self.confirm_days, len(df)):
                if i - self.confirm_days >= 0 and df['cross_change'].iloc[i-self.confirm_days] > 0:
                    # Potential bullish crossover
                    if all(df['sma_cross'].iloc[i-self.confirm_days:i+1] == 1):
                        df.loc[df.index[i], 'signal'] = 1
                elif i - self.confirm_days >= 0 and df['cross_change'].iloc[i-self.confirm_days] < 0:
                    # Potential bearish crossover
                    if all(df['sma_cross'].iloc[i-self.confirm_days:i+1] == -1):
                        df.loc[df.index[i], 'signal'] = -1
        else:
            # No confirmation period, generate signals directly on crossovers
            df.loc[df['cross_change'] > 0, 'signal'] = 1  # Bullish crossover
            df.loc[df['cross_change'] < 0, 'signal'] = -1  # Bearish crossover
        
        # Apply noise filtering if enabled
        if self.filter_noise:
            # Add additional constraints to avoid false signals
            
            # Calculate price volatility
            df['volatility'] = df['close'].pct_change().rolling(window=20).std()
            
            # Calculate distance between MAs as percentage
            df['ma_gap_pct'] = 100 * (df['short_ma'] - df['long_ma']) / df['long_ma']
            
            # Filter out signals where the MA gap is too small relative to volatility
            # A small gap may indicate a false or weak signal
            volatility_threshold = df['volatility'] * 100  # Convert to percentage
            
            # Only keep signals where the gap is significant enough
            for i in range(len(df)):
                if df['signal'].iloc[i] == 1:
                    if df['ma_gap_pct'].iloc[i] <= volatility_threshold.iloc[i]:
                        df.loc[df.index[i], 'signal'] = 0
                elif df['signal'].iloc[i] == -1:
                    if abs(df['ma_gap_pct'].iloc[i]) <= volatility_threshold.iloc[i]:
                        df.loc[df.index[i], 'signal'] = 0
        
        # Generate position changes
        df['position'] = df['signal'] * 2  # Multiply by 2 to match the expected format
        
        # Fill NaN values
        df['position'] = df['position'].fillna(0)
        
        # Count signals
        buy_signals = (df['position'] == 2).sum()
        sell_signals = (df['position'] == -2).sum()
        
        logger.info(f"Generated {buy_signals} buy signals and {sell_signals} sell signals")
        
        # Clean up intermediate columns if not needed
        for col in ['cross_change', 'sma_cross']:
            if col in df.columns:
                df = df.drop(col, axis=1)
                
        if not self.filter_noise:
            for col in ['volatility', 'ma_gap_pct']:
                if col in df.columns:
                    df = df.drop(col, axis=1)
        
        return df