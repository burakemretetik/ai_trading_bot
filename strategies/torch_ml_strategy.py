# strategies/torch_ml_strategy.py - PyTorch-based trading strategy
import pandas as pd
import numpy as np
from models.torch_price_predictor import TorchPricePredictor
import logging

logger = logging.getLogger(__name__)

class TorchMLStrategy:
    """Trading strategy based on PyTorch predictions."""
    
    def __init__(self, threshold=0.001, retrain_period=30, hidden_dim=64, num_layers=2, dropout=0.2):
        """
        Initialize PyTorch ML strategy.
        
        Args:
            threshold (float): Minimum predicted return to enter a position
            retrain_period (int): Number of days before retraining the model
            hidden_dim (int): Number of neurons in hidden layers
            num_layers (int): Number of hidden layers
            dropout (float): Dropout rate for regularization
        """
        self.predictor = TorchPricePredictor(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )
        self.threshold = threshold
        self.retrain_period = retrain_period
        self.last_trained = None
    
    def train_model(self, data):
        """Train the ML model on historical data."""
        logger.info("Training PyTorch ML model for trading strategy...")
        self.predictor.train(data)
        self.last_trained = data.index[-1]
    
    def generate_signals(self, data):
        """Generate buy/sell signals based on ML predictions."""
        logger.info("Generating signals using PyTorch ML strategy...")
        
        # Make a copy of the data
        df = data.copy()
        
        # Train or retrain the model if needed
        if self.last_trained is None or (
            df.index[-1] - self.last_trained).days >= self.retrain_period:
            self.train_model(df)
        
        # Get predictions
        predictions = self.predictor.predict(df)
        df = df.join(predictions[['predicted_return']], how='left')
        
        # Initialize signals to 0
        df['signal'] = 0
        
        # Generate signals based on predicted returns
        df.loc[df['predicted_return'] > self.threshold, 'signal'] = 1  # Buy
        df.loc[df['predicted_return'] < -self.threshold, 'signal'] = -1  # Sell
        
        # Generate position changes
        df['position'] = df['signal'].diff()
        
        # Drop NaN values
        df.dropna(inplace=True)
        
        logger.info(f"Generated {len(df[df['position'] == 2])} buy signals and {len(df[df['position'] == -2])} sell signals")
        
        return df