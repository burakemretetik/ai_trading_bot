# models/torch_price_predictor.py - PyTorch model for price prediction
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import logging
import os
import joblib

logger = logging.getLogger(__name__)

class PricePredictionModel(nn.Module):
    """Neural network model for price prediction."""
    
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.2):
        """Initialize the neural network model.
        
        Args:
            input_dim (int): Number of input features
            hidden_dim (int): Number of neurons in hidden layers
            num_layers (int): Number of hidden layers
            dropout (float): Dropout rate for regularization
        """
        super(PricePredictionModel, self).__init__()
        
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
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, 1))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        """Forward pass through the network."""
        return self.model(x)


class TorchPricePredictor:
    """Class for predicting price movements using PyTorch."""
    
    def __init__(self, hidden_dim=64, num_layers=2, dropout=0.2, learning_rate=0.001, batch_size=32, epochs=100):
        """Initialize the price predictor with model parameters."""
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        
        self.model = None
        self.scaler = StandardScaler()
        self.features = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
    
    def _create_features(self, df):
        """Create technical indicators as features."""
        data = df.copy()
        
        # Log the initial data shape
        logger.info(f"Creating features for {len(data)} data points")
        if len(data) < 100:
            logger.warning("Dataset may be too small for reliable feature creation")
        
        # Fill any existing NaN values in input data
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in data.columns and data[col].isna().any():
                logger.warning(f"Found NaN values in {col} column, filling with forward fill")
                data[col] = data[col].fillna(method='ffill')
        
        # Price-based features
        data['returns'] = data['close'].pct_change()
        data['log_returns'] = np.log(data['close'] / data['close'].shift(1))
        
        # Moving averages - use smaller windows if data is limited
        ma_windows = [5, 10, 20, 50]
        if len(data) >= 100:
            ma_windows.append(100)
        
        for window in ma_windows:
            if len(data) >= window:
                data[f'ma_{window}'] = data['close'].rolling(window=window, min_periods=1).mean()
                data[f'ma_ratio_{window}'] = data['close'] / data[f'ma_{window}']
        
        # Volatility
        vol_windows = [5, 10, 20]
        for window in vol_windows:
            if len(data) >= window:
                data[f'volatility_{window}'] = data['returns'].rolling(window=window, min_periods=1).std()
        
        # RSI (Relative Strength Index)
        if len(data) >= 14:
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
            # Avoid division by zero
            loss = loss.replace(0, np.nan)
            rs = gain / loss
            # Fill NaN values with 0.5 (neutral RSI)
            rs = rs.fillna(1) 
            data['rsi_14'] = 100 - (100 / (1 + rs))
        
        # MACD - only calculate if we have enough data
        if len(data) >= 26:
            data['ema_12'] = data['close'].ewm(span=12, adjust=False, min_periods=1).mean()
            data['ema_26'] = data['close'].ewm(span=26, adjust=False, min_periods=1).mean()
            data['macd'] = data['ema_12'] - data['ema_26']
            data['macd_signal'] = data['macd'].ewm(span=9, adjust=False, min_periods=1).mean()
            data['macd_hist'] = data['macd'] - data['macd_signal']
        
        # Volume-based features
        if 'volume' in data.columns and len(data) >= 5:
            data['volume_ma_5'] = data['volume'].rolling(window=5, min_periods=1).mean()
            # Avoid division by zero
            data['volume_ma_5'] = data['volume_ma_5'].replace(0, np.nan)
            data['volume_ratio'] = data['volume'] / data['volume_ma_5']
            data['volume_ratio'] = data['volume_ratio'].fillna(1)
        
        # Target variable: future returns (next day's return)
        data['target'] = data['returns'].shift(-1)
        
        # Fill remaining NaN values with 0 or appropriate values
        for col in data.columns:
            if data[col].isna().any():
                nan_count = data[col].isna().sum()
                logger.warning(f"Column {col} has {nan_count} NaN values, filling with appropriate values")
                
                if col in ['returns', 'log_returns', 'target']:
                    # Fill with 0 (no change)
                    data[col] = data[col].fillna(0)
                elif 'ratio' in col:
                    # Fill with 1 (no change)
                    data[col] = data[col].fillna(1)
                elif 'rsi' in col:
                    # Fill with 50 (neutral)
                    data[col] = data[col].fillna(50)
                else:
                    # Fill with mean
                    data[col] = data[col].fillna(data[col].mean())
        
        logger.info(f"Feature creation complete. Data shape after feature creation: {data.shape}")
        
        return data

    def prepare_data(self, df):
        """Prepare data for training or prediction."""
        # Check if dataframe is empty
        if df.empty:
            logger.error("Empty dataframe provided for feature creation")
            return pd.DataFrame()
        
        # Create features
        data = self._create_features(df)
        
        # Define features to use (skip missing columns)
        all_potential_features = [
            'returns', 'log_returns', 
            'ma_5', 'ma_10', 'ma_20', 'ma_50', 'ma_100',
            'ma_ratio_5', 'ma_ratio_10', 'ma_ratio_20', 'ma_ratio_50', 'ma_ratio_100',
            'volatility_5', 'volatility_10', 'volatility_20',
            'rsi_14', 'ema_12', 'ema_26', 'macd', 'macd_signal', 'macd_hist',
            'volume_ma_5', 'volume_ratio'
        ]
        
        # Only include features that exist in the data
        self.features = [col for col in all_potential_features if col in data.columns]
        
        logger.info(f"Selected {len(self.features)} features: {self.features}")
        
        # Make sure we have data
        if len(data) == 0:
            logger.error("No data left after feature creation")
        
        return data
    
    def train(self, df, test_size=0.2):
        """Train the model on historical data."""
        logger.info("Training PyTorch price prediction model...")
        
        # Prepare data
        data = self.prepare_data(df)
        
        # Make sure we have enough data to train
        if len(data) < 100:
            logger.warning(f"Training with only {len(data)} samples, model may not perform well")
            if len(data) < 30:
                logger.error("Not enough data for training, need at least 30 samples")
                return None
        
        # Split features and target
        X = data[self.features].values
        y = data['target'].values.reshape(-1, 1)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split into training and testing sets
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
        train_loader = DataLoader(train_dataset, batch_size=min(self.batch_size, len(X_train)), shuffle=True)
        
        # Initialize model
        self.model = PricePredictionModel(
            input_dim=X_train.shape[1],
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)
        
        # Define loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
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
            
            # Print progress every 10 epochs
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch [{epoch+1}/{self.epochs}], Loss: {total_loss/len(train_loader):.6f}")
        
        # Evaluate on test set
        self.model.eval()
        with torch.no_grad():
            train_pred = self.model(X_train_tensor)
            test_pred = self.model(X_test_tensor)
            
            train_loss = criterion(train_pred, y_train_tensor).item()
            test_loss = criterion(test_pred, y_test_tensor).item()
            
            # Calculate R² score (coefficient of determination)
            y_train_mean = torch.mean(y_train_tensor)
            y_test_mean = torch.mean(y_test_tensor)
            
            ss_tot_train = torch.sum((y_train_tensor - y_train_mean) ** 2)
            ss_res_train = torch.sum((y_train_tensor - train_pred) ** 2)
            r2_train = 1 - ss_res_train / ss_tot_train
            
            ss_tot_test = torch.sum((y_test_tensor - y_test_mean) ** 2)
            ss_res_test = torch.sum((y_test_tensor - test_pred) ** 2)
            r2_test = 1 - ss_res_test / ss_tot_test
        
        logger.info(f"Model trained. MSE (train): {train_loss:.6f}, MSE (test): {test_loss:.6f}")
        logger.info(f"R² score (train): {r2_train.item():.4f}, R² score (test): {r2_test.item():.4f}")
        
        # Feature importance (approximated by feature weights)
        feature_importance = self._calculate_feature_importance()
        
        logger.info(f"Top 5 important features: {feature_importance.head(5)['feature'].tolist()}")
        
        return {
            'train_loss': train_loss,
            'test_loss': test_loss,
            'r2_train': r2_train.item(),
            'r2_test': r2_test.item(),
            'feature_importance': feature_importance
        }
    
    def _calculate_feature_importance(self):
        """Calculate feature importance based on weights of the first layer."""
        if self.model is None:
            return pd.DataFrame(columns=['feature', 'importance'])
        
        # Get weights from the first layer
        weights = self.model.model[0].weight.detach().cpu().numpy()
        
        # Take absolute values and average across neurons
        importance = np.abs(weights).mean(axis=0)
        
        # Create feature importance DataFrame
        feature_importance = pd.DataFrame({
            'feature': self.features,
            'importance': importance
        })
        
        # Sort by importance
        feature_importance = feature_importance.sort_values('importance', ascending=False)
        
        return feature_importance
    
    def predict(self, df):
        """Predict future returns using the trained model."""
        if self.model is None:
            logger.error("Model not trained yet.")
            return None
        
        # Prepare data
        data = self.prepare_data(df)
        
        if data.empty:
            logger.error("No valid data for prediction after preprocessing")
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
        
        return data[['close', 'predicted_return', 'predicted_price']]
    
    def save_model(self, filepath='models/torch_price_predictor.pth'):
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
                'dropout': self.dropout
            }, metadata_path)
            
            logger.info(f"Model saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
    
    def load_model(self, filepath='models/torch_price_predictor.pth'):
        """Load a trained model from a file."""
        try:
            # Load metadata
            metadata_path = filepath.replace('.pth', '_metadata.joblib')
            metadata = joblib.load(metadata_path)
            
            self.scaler = metadata['scaler']
            self.features = metadata['features']
            self.hidden_dim = metadata['hidden_dim']
            self.num_layers = metadata['num_layers']
            self.dropout = metadata['dropout']
            
            # Initialize model
            input_dim = len(self.features)
            self.model = PricePredictionModel(
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
            logger.error(f"Error loading model: {e}")
            return False