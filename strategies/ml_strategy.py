# strategies/ml_strategy.py - Improved ML-based strategy
import pandas as pd
import numpy as np
import logging
from typing import Optional, Dict, Any
from strategies.strategy import Strategy
from models.predictor import PricePredictor

logger = logging.getLogger(__name__)

class MLStrategy(Strategy):
    """Trading strategy based on machine learning predictions with improved validation."""
    
    def __init__(self, threshold=0.001, retrain_period=30, hidden_dim=64, num_layers=2, 
                 dropout=0.2, feature_engineering='basic', stop_loss=0.05):
        """
        Initialize ML strategy.
        
        Args:
            threshold: Minimum predicted return to enter a position
            retrain_period: Number of days before retraining the model
            hidden_dim: Hidden dimension size for neural network
            num_layers: Number of hidden layers
            dropout: Dropout rate
            feature_engineering: Feature engineering level ('basic', 'advanced')
            stop_loss: Stop loss percentage
        """
        super().__init__("ML")
        self.predictor = PricePredictor(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            feature_engineering=feature_engineering
        )
        self.threshold = threshold
        self.retrain_period = retrain_period
        self.last_trained = None
        self.stop_loss = stop_loss
        self.min_confidence = 0.6  # Minimum prediction confidence to take a trade
    
    def train_model(self, data: pd.DataFrame) -> bool:
        """
        Train the ML model on historical data.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Boolean indicating if training was successful
        """
        logger.info("Training ML model for trading strategy...")
        
        # Validate input data
        if self._validate_input_data(data) is False:
            return False
        
        # Train the model
        training_result = self.predictor.train(data)
        
        if training_result is None:
            logger.error("Model training failed")
            return False
        
        # Log training metrics
        logger.info(f"Model trained with test loss: {training_result['test_loss']:.6f}")
        logger.info(f"Direction accuracy: {training_result.get('direction_accuracy', 0):.2f}")
        
        self.last_trained = data.index[-1]
        return True
    
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
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return False
        
        # Check for sufficient data points
        min_required = 50  # Need at least 50 data points for meaningful training
        if len(data) < min_required:
            logger.error(f"Insufficient data points: {len(data)}. Need at least {min_required}.")
            return False
        
        # Check for NaN values
        if data[required_columns].isna().any().any():
            logger.warning("Data contains NaN values. These will be handled during feature processing.")
        
        return True
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate buy/sell signals based on ML predictions.
        
        Args:
            data: DataFrame with price data
            
        Returns:
            DataFrame with signals
        """
        logger.info(f"Generating signals using ML strategy with threshold {self.threshold}")
        
        # Validate input data
        if self._validate_input_data(data) is False:
            # Return original dataframe with empty signals
            df = data.copy()
            df['signal'] = 0
            df['position'] = 0
            logger.error("Cannot generate signals due to invalid input data")
            return df
        
        # Make a copy of the data
        df = data.copy()
        
        # Train or retrain the model if needed
        model_trained = False
        if self.last_trained is None:
            logger.info("First-time model training")
            model_trained = self.train_model(df)
        elif (df.index[-1] - self.last_trained).days >= self.retrain_period:
            logger.info(f"Retraining model after {self.retrain_period} days")
            model_trained = self.train_model(df)
        else:
            model_trained = True  # Already trained and within retrain period
        
        if not model_trained:
            logger.error("Model training/retraining failed")
            # Return original dataframe with empty signals
            df['signal'] = 0
            df['position'] = 0
            return df
        
        # Get predictions
        predictions = self.predictor.predict(df)
        if predictions is None:
            logger.error("Failed to get predictions")
            # Return original dataframe with empty signals
            df['signal'] = 0
            df['position'] = 0
            return df
            
        # Merge predictions with original data
        for col in predictions.columns:
            if col not in df.columns:
                df[col] = predictions[col]
        
        # Check if necessary prediction columns exist
        required_pred_columns = ['predicted_return', 'prediction_confidence']
        if not all(col in df.columns for col in required_pred_columns):
            logger.error(f"Missing required prediction columns: {[col for col in required_pred_columns if col not in df.columns]}")
            df['signal'] = 0
            df['position'] = 0
            return df
            
        # Initialize signals to 0
        df['signal'] = 0
        
        # Generate signals based on predicted returns AND confidence
        df.loc[(df['predicted_return'] > self.threshold) & 
               (df['prediction_confidence'] >= self.min_confidence), 'signal'] = 1  # Buy
               
        df.loc[(df['predicted_return'] < -self.threshold) & 
               (df['prediction_confidence'] >= self.min_confidence), 'signal'] = -1  # Sell
        
        # Calculate positions (current signal)
        # This represents the position we want to be in, not the change in position
        
        # Generate position changes
        df['position'] = df['signal'].diff()
        
        # Fill NaN values
        df['position'] = df['position'].fillna(0)
        
        # Apply stop loss to limit losses (simulation)
        self._apply_stop_loss(df)
        
        # Count signals
        buy_signals = (df['position'] > 0).sum()
        sell_signals = (df['position'] < 0).sum()
        
        logger.info(f"Generated {buy_signals} buy signals and {sell_signals} sell signals")
        
        return df
    
    def _apply_stop_loss(self, df: pd.DataFrame) -> None:
        """
        Apply stop loss logic to the signals dataframe.
        
        This simulates what would happen if stop losses were applied.
        
        Args:
            df: DataFrame with signals to modify in-place
        """
        # Need to track positions and entry prices
        entry_price = None
        in_position = False  # 1 for long, -1 for short, 0 for no position
        position_type = 0
        
        for i in range(1, len(df)):
            prev_idx = df.index[i-1]
            curr_idx = df.index[i]
            
            # Check if we're entering a position
            if df.at[curr_idx, 'position'] == 2:  # Entering long
                entry_price = df.at[curr_idx, 'close']
                in_position = True
                position_type = 1
            elif df.at[curr_idx, 'position'] == -2:  # Entering short
                entry_price = df.at[curr_idx, 'close']
                in_position = True
                position_type = -1
            
            # Check for stop loss if we're in a position
            if in_position and entry_price is not None:
                current_price = df.at[curr_idx, 'close']
                
                # For long positions, check if price fell below stop loss
                if position_type == 1 and current_price < entry_price * (1 - self.stop_loss):
                    # Trigger stop loss - exit position
                    df.at[curr_idx, 'position'] = -2  # Force exit from long
                    df.at[curr_idx, 'signal'] = 0  # Reset signal
                    in_position = False
                    entry_price = None
                    position_type = 0
                    logger.debug(f"Stop loss triggered at {curr_idx} for long position")
                
                # For short positions, check if price rose above stop loss
                elif position_type == -1 and current_price > entry_price * (1 + self.stop_loss):
                    # Trigger stop loss - exit position
                    df.at[curr_idx, 'position'] = 2  # Force exit from short
                    df.at[curr_idx, 'signal'] = 0  # Reset signal
                    in_position = False
                    entry_price = None
                    position_type = 0
                    logger.debug(f"Stop loss triggered at {curr_idx} for short position")
            
            # Check if we're exiting a position
            if position_type == 1 and df.at[curr_idx, 'position'] == -2:  # Exiting long
                in_position = False
                entry_price = None
                position_type = 0
            elif position_type == -1 and df.at[curr_idx, 'position'] == 2:  # Exiting short
                in_position = False
                entry_price = None
                position_type = 0