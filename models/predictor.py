# models/predictor.py - Improved price prediction model
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
import logging
import os
import joblib
from datetime import datetime
from typing import Dict, Any, Union, Optional, Tuple, List

logger = logging.getLogger(__name__)

class PriceModel(nn.Module):
    """Neural network model for price prediction with improved architecture."""
    
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.2):
        """Initialize the neural network model."""
        super(PriceModel, self).__init__()
        
        layers = []
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        # Output layer - no activation since we're predicting a continuous value
        # This is a regression problem, so we want raw values
        layers.append(nn.Linear(hidden_dim, 1))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        """Forward pass through the network."""
        return self.model(x)


class PricePredictor:
    """Price prediction model using PyTorch with improved features and safeguards."""
    
    def __init__(self, hidden_dim=64, num_layers=2, dropout=0.2, learning_rate=0.001, 
                 batch_size=32, epochs=100, feature_engineering='basic'):
        """
        Initialize the price predictor with model parameters.
        
        Args:
            hidden_dim: Size of hidden layers
            num_layers: Number of hidden layers
            dropout: Dropout rate for regularization
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training
            epochs: Number of training epochs
            feature_engineering: Feature engineering level ('basic', 'advanced')
        """
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.feature_engineering = feature_engineering
        
        self.model = None
        self.scaler = RobustScaler()  # Using RobustScaler for better handling of outliers
        self.features = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
    
    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create technical indicators and prepare features with improved handling of edge cases.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added features
        """
        # Validate input data
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in df.columns:
                logger.error(f"Required column '{col}' missing from input data")
                return pd.DataFrame()
        
        # Make a copy of the data
        data = df.copy()
        
        # Handle potential zero or negative values
        eps = 1e-8  # Small epsilon to prevent division by zero
        for col in ['open', 'high', 'low', 'close']:
            data[col] = data[col].clip(lower=eps)  # Ensure all values are positive
        
        try:
            # Basic returns
            data['returns'] = data['close'].pct_change().fillna(0)
            
            # Use log returns for better statistical properties
            # Safely handle log calculation for price ratios
            data['log_returns'] = np.log(data['close'] / data['close'].shift(1).clip(lower=eps)).fillna(0)
            
            # Moving averages with minimum periods set properly
            for window in [5, 10, 20, 50]:
                min_periods = min(window, len(data) // 2)  # Adjust min_periods based on data length
                data[f'ma_{window}'] = data['close'].rolling(window=window, min_periods=min_periods).mean().fillna(method='bfill').fillna(data['close'])
                data[f'ma_ratio_{window}'] = (data['close'] / data[f'ma_{window}']).fillna(1)
            
            # Volatility with proper window size adjustment
            vol_window = min(10, len(data) // 2)
            data['volatility_10'] = data['returns'].rolling(window=vol_window, min_periods=2).std().fillna(0)
            
            # RSI (Relative Strength Index) with safer calculations
            rsi_window = min(14, len(data) // 2)
            delta = data['close'].diff().fillna(0)
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            
            avg_gain = gain.rolling(window=rsi_window, min_periods=1).mean()
            avg_loss = loss.rolling(window=rsi_window, min_periods=1).mean()
            
            # Safe division avoiding zeros
            rs = avg_gain / avg_loss.replace(0, np.nan).fillna(avg_gain).clip(lower=eps)
            data['rsi_14'] = 100 - (100 / (1 + rs))
            
            # Add advanced features if requested
            if self.feature_engineering == 'advanced':
                # Add MACD
                data['ema_12'] = data['close'].ewm(span=12, adjust=False).mean()
                data['ema_26'] = data['close'].ewm(span=26, adjust=False).mean()
                data['macd'] = data['ema_12'] - data['ema_26']
                data['macd_signal'] = data['macd'].ewm(span=9, adjust=False).mean()
                data['macd_hist'] = data['macd'] - data['macd_signal']
                
                # Add Bollinger Bands
                bb_window = min(20, len(data) // 2)
                data['bb_middle'] = data['close'].rolling(window=bb_window, min_periods=min(5, len(data) // 4)).mean()
                data['bb_std'] = data['close'].rolling(window=bb_window, min_periods=min(5, len(data) // 4)).std()
                data['bb_upper'] = data['bb_middle'] + 2 * data['bb_std']
                data['bb_lower'] = data['bb_middle'] - 2 * data['bb_std']
                data['bb_width'] = (data['bb_upper'] - data['bb_lower']) / data['bb_middle']
                data['bb_pct'] = (data['close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'] + eps)
                
                # Add lagged features
                for lag in [1, 2, 3, 5]:
                    data[f'lag_returns_{lag}'] = data['returns'].shift(lag).fillna(0)
                
                # Volume indicators
                data['volume_ratio'] = data['volume'] / data['volume'].rolling(window=10, min_periods=1).mean().fillna(data['volume'])
                
                # Price patterns
                for window in [5, 10]:
                    data[f'high_diff_{window}'] = data['high'] / data['high'].rolling(window=window, min_periods=1).max().fillna(data['high']) - 1
                    data[f'low_diff_{window}'] = data['low'] / data['low'].rolling(window=window, min_periods=1).min().fillna(data['low']) - 1
            
            # Target variable: future returns (next day's return)
            data['target'] = data['returns'].shift(-1)
            
            # Fill NaN values for each column with appropriate defaults
            # For trend indicators, fill with neutral value
            for col in data.columns:
                if col in ['target']:
                    continue  # Skip target as we'll handle this separately
                if col.startswith('rsi'):
                    data[col] = data[col].fillna(50)  # Neutral RSI
                elif col.startswith('ma_ratio'):
                    data[col] = data[col].fillna(1)  # Price at MA
                elif col.startswith('bb_pct'):
                    data[col] = data[col].fillna(0.5)  # Middle of BB
                elif 'returns' in col or 'volatility' in col:
                    data[col] = data[col].fillna(0)  # No change/volatility
                else:
                    data[col] = data[col].fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            # Define features to use based on feature engineering level
            if self.feature_engineering == 'advanced':
                self.features = [
                    'returns', 'log_returns',
                    'ma_ratio_5', 'ma_ratio_10', 'ma_ratio_20', 'ma_ratio_50',
                    'volatility_10', 'rsi_14',
                    'macd', 'macd_hist',
                    'bb_width', 'bb_pct',
                    'lag_returns_1', 'lag_returns_3',
                    'volume_ratio',
                    'high_diff_5', 'low_diff_5'
                ]
            else:
                self.features = [
                    'returns', 'log_returns',
                    'ma_ratio_5', 'ma_ratio_10', 'ma_ratio_20', 'ma_ratio_50',
                    'volatility_10', 'rsi_14'
                ]
            
            # Fill target NaN values with zeros (at the end of the dataset)
            data['target'] = data['target'].fillna(0)
            
            return data
        except Exception as e:
            logger.error(f"Error in feature preparation: {str(e)}")
            return pd.DataFrame()
    
    def train(self, df: pd.DataFrame, test_size: float = 0.2, validation: bool = True) -> Dict[str, Any]:
        """
        Train the model on historical data with improved validation.
        
        Args:
            df: DataFrame with OHLCV data
            test_size: Proportion of data to use for testing
            validation: Whether to use time series cross-validation
            
        Returns:
            Dictionary with training results
        """
        logger.info("Training price prediction model...")
        
        try:
            # Prepare data with features
            data = self._prepare_features(df)
            
            if data.empty:
                logger.error("Feature preparation failed")
                return None
            
            # Make sure we have enough data
            min_samples = max(30, self.batch_size * 3)
            if len(data) < min_samples:
                logger.error(f"Not enough data for training, need at least {min_samples} samples")
                return None
            
            # Split features and target
            X = data[self.features].values
            y = data['target'].values.reshape(-1, 1)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            if validation:
                # Use time series cross-validation
                tscv = TimeSeriesSplit(n_splits=5)
                val_scores = []
                
                for train_idx, val_idx in tscv.split(X_scaled):
                    # Split data
                    X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
                    y_train, y_val = y[train_idx], y[val_idx]
                    
                    # Convert to PyTorch tensors
                    X_train_tensor = torch.FloatTensor(X_train).to(self.device)
                    y_train_tensor = torch.FloatTensor(y_train).to(self.device)
                    X_val_tensor = torch.FloatTensor(X_val).to(self.device)
                    y_val_tensor = torch.FloatTensor(y_val).to(self.device)
                    
                    # Initialize model for this fold
                    fold_model = PriceModel(
                        input_dim=X_train.shape[1],
                        hidden_dim=self.hidden_dim,
                        num_layers=self.num_layers,
                        dropout=self.dropout
                    ).to(self.device)
                    
                    # Define loss function and optimizer
                    criterion = nn.MSELoss()
                    optimizer = optim.Adam(fold_model.parameters(), lr=self.learning_rate)
                    
                    # Training loop
                    fold_model.train()
                    for epoch in range(self.epochs):
                        # Forward pass
                        outputs = fold_model(X_train_tensor)
                        loss = criterion(outputs, y_train_tensor)
                        
                        # Backward pass and optimize
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    
                    # Evaluate on validation set
                    fold_model.eval()
                    with torch.no_grad():
                        val_pred = fold_model(X_val_tensor)
                        val_loss = criterion(val_pred, y_val_tensor).item()
                        val_scores.append(val_loss)
                
                # Use the average validation score
                test_loss = np.mean(val_scores)
                
                # Train final model on all data
                X_train_tensor = torch.FloatTensor(X_scaled).to(self.device)
                y_train_tensor = torch.FloatTensor(y).to(self.device)
            else:
                # Split into training and testing sets for traditional validation
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y, test_size=test_size, shuffle=False
                )
                
                # Convert to PyTorch tensors
                X_train_tensor = torch.FloatTensor(X_train).to(self.device)
                y_train_tensor = torch.FloatTensor(y_train).to(self.device)
                X_test_tensor = torch.FloatTensor(X_test).to(self.device)
                y_test_tensor = torch.FloatTensor(y_test).to(self.device)
            
            # Create DataLoader for batching
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            batch_size = min(self.batch_size, len(X_train_tensor))
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            
            # Initialize model
            self.model = PriceModel(
                input_dim=X_scaled.shape[1],
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
                dropout=self.dropout
            ).to(self.device)
            
            # Define loss function and optimizer
            criterion = nn.MSELoss()
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            
            # Early stopping parameters
            patience = 10
            best_loss = float('inf')
            patience_counter = 0
            
            # Training loop
            self.model.train()
            for epoch in range(self.epochs):
                total_loss = 0
                for batch_X, batch_y in train_loader:
                    # Forward pass
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    
                    # Backward pass and optimize
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                
                avg_loss = total_loss / len(train_loader)
                
                # Early stopping check
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logger.info(f"Early stopping at epoch {epoch+1}")
                        break
                
                # Print progress every 20 epochs
                if (epoch + 1) % 20 == 0:
                    logger.info(f"Epoch [{epoch+1}/{self.epochs}], Loss: {avg_loss:.6f}")
            
            # Evaluate on test set if not using validation
            if not validation:
                self.model.eval()
                with torch.no_grad():
                    test_pred = self.model(X_test_tensor)
                    test_loss = criterion(test_pred, y_test_tensor).item()
            
            logger.info(f"Model trained. Test MSE: {test_loss:.6f}")
            
            # Compute additional metrics
            self.model.eval()
            with torch.no_grad():
                # Use full data to get baseline metrics
                full_pred = self.model(torch.FloatTensor(X_scaled).to(self.device)).cpu().numpy().flatten()
                full_pred_df = pd.DataFrame({
                    'actual': y.flatten(),
                    'predicted': full_pred
                })
                
                # Direction accuracy (up/down)
                direction_correct = np.sum((full_pred_df['actual'] > 0) == (full_pred_df['predicted'] > 0))
                direction_accuracy = direction_correct / len(full_pred_df)
                
                # Use correlation as another metric
                correlation = full_pred_df.corr().iloc[0, 1]
            
            return {
                'test_loss': test_loss,
                'direction_accuracy': direction_accuracy,
                'correlation': correlation,
                'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'epochs_trained': epoch + 1
            }
            
        except Exception as e:
            logger.error(f"Error during model training: {str(e)}")
            return None
    
    def predict(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Predict future returns using the trained model with better validation.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with predictions or None if prediction fails
        """
        if self.model is None:
            logger.error("Model not trained yet.")
            return None
        
        try:
            # Prepare data with features
            data = self._prepare_features(df)
            
            if data.empty:
                logger.error("Feature preparation failed")
                return None
            
            # Check if we have all required features
            for feature in self.features:
                if feature not in data.columns:
                    logger.error(f"Required feature '{feature}' missing from data")
                    return None
            
            # Extract features
            X = data[self.features].values
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Convert to PyTorch tensor
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            
            # Make predictions
            self.model.eval()
            with torch.no_grad():
                predictions = self.model(X_tensor).cpu().numpy().flatten()
            
            # Add predictions to dataframe
            data['predicted_return'] = predictions
            data['predicted_price'] = data['close'] * (1 + data['predicted_return'])
            
            # Calculate prediction confidence based on model architecture
            # Simple approach: based on deviation from historical volatility
            historical_std = data['returns'].std()
            data['prediction_confidence'] = 1.0 - np.minimum(1.0, np.abs(data['predicted_return']) / (3 * historical_std))
            
            # Return only the necessary columns
            result = data[['close', 'predicted_return', 'predicted_price', 'prediction_confidence']]
            
            return result
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            return None
    
    def save_model(self, filepath='models/price_predictor.pth'):
        """Save the trained model to a file."""
        if self.model is None:
            logger.error("No trained model to save.")
            return False
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Save model state dict
            torch.save(self.model.state_dict(), filepath)
            
            # Save scaler and features
            metadata_path = filepath.replace('.pth', '_metadata.joblib')
            joblib.dump({
                'scaler': self.scaler,
                'features': self.features,
                'hidden_dim': self.hidden_dim,
                'num_layers': self.num_layers,
                'dropout': self.dropout,
                'feature_engineering': self.feature_engineering,
                'save_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }, metadata_path)
            
            logger.info(f"Model saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            return False
    
    def load_model(self, filepath='models/price_predictor.pth'):
        """Load a trained model from a file."""
        try:
            # Check if file exists
            if not os.path.exists(filepath):
                logger.error(f"Model file not found: {filepath}")
                return False
                
            # Load metadata
            metadata_path = filepath.replace('.pth', '_metadata.joblib')
            if not os.path.exists(metadata_path):
                logger.error(f"Model metadata file not found: {metadata_path}")
                return False
                
            metadata = joblib.load(metadata_path)
            
            # Load model parameters from metadata
            self.scaler = metadata['scaler']
            self.features = metadata['features']
            self.hidden_dim = metadata['hidden_dim']
            self.num_layers = metadata['num_layers']
            self.dropout = metadata['dropout']
            
            # Load feature engineering setting if available (for backward compatibility)
            if 'feature_engineering' in metadata:
                self.feature_engineering = metadata['feature_engineering']
            
            # Initialize model
            input_dim = len(self.features)
            self.model = PriceModel(
                input_dim=input_dim,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
                dropout=self.dropout
            ).to(self.device)
            
            # Load model state dict
            self.model.load_state_dict(torch.load(filepath, map_location=self.device))
            self.model.eval()
            
            logger.info(f"Model loaded from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False