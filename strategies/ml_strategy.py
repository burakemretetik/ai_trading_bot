# strategies/ml_strategy.py - ML-based strategy
import pandas as pd
import numpy as np
import logging
from strategies.strategy import Strategy
from models.predictor import PricePredictor

logger = logging.getLogger(__name__)

class MLStrategy(Strategy):
    """Trading strategy based on machine learning predictions."""
    
    def __init__(self, threshold=0.001, retrain_period=30, hidden_dim=64, num_layers=2, dropout=0.2):
        """
        Initialize ML strategy.
        
        Args:
            threshold: Minimum predicted return to enter a position
            retrain_period: Number of days before retraining the model
            hidden_dim: Hidden dimension size for neural network
            num_layers: Number of hidden layers
            dropout: Dropout rate
        """
        super().__init__("ML")
        self.predictor = PricePredictor(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )
        self.threshold = threshold
        self.retrain_period = retrain_period
        self.last_trained = None
    
    def train_model(self, data):
        """Train the ML model on historical data."""
        logger.info("Training ML model for trading strategy...")
        self.predictor.train(data)
        self.last_trained = data.index[-1]
    
    def generate_signals(self, data):
        """
        Generate buy/sell signals based on ML predictions.
        
        Args:
            data: DataFrame with price data
            
        Returns:
            DataFrame with signals
        """
        logger.info(f"Generating signals using ML strategy with threshold {self.threshold}")
        
        # Make a copy of the data
        df = data.copy()
        
        # Train or retrain the model if needed
        if self.last_trained is None:
            self.train_model(df)
        elif (df.index[-1] - self.last_trained).days >= self.retrain_period:
            logger.info(f"Retraining model after {self.retrain_period} days")
            self.train_model(df)
        
        # Get predictions
        predictions = self.predictor.predict(df)
        if predictions is None:
            logger.error("Failed to get predictions")
            # Return original dataframe with empty signals
            df['signal'] = 0
            df['position'] = 0
            return df
            
        # Merge predictions with original data
        df = df.join(predictions[['predicted_return']], how='left')
        
        # Initialize signals to 0
        df['signal'] = 0
        
        # Generate signals based on predicted returns
        df.loc[df['predicted_return'] > self.threshold, 'signal'] = 1  # Buy
        df.loc[df['predicted_return'] < -self.threshold, 'signal'] = -1  # Sell
        
        # Generate position changes
        df['position'] = df['signal'].diff()
        
        # Fill NaN values
        df['position'] = df['position'].fillna(0)
        
        # Count signals
        buy_signals = (df['position'] == 2).sum()
        sell_signals = (df['position'] == -2).sum()
        
        logger.info(f"Generated {buy_signals} buy signals and {sell_signals} sell signals")
        
        return df