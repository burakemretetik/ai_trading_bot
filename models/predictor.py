# models/predictor.py - Simplified price prediction model
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
from datetime import datetime

logger = logging.getLogger(__name__)

class PriceModel(nn.Module):
    """Neural network model for price prediction."""
    
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
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, 1))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        """Forward pass through the network."""
        return self.model(x)


class PricePredictor:
    """Price prediction model using PyTorch."""
    
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
    
    def _prepare_features(self, df):
        """Create technical indicators and prepare features."""
        # Make a copy of the data
        data = df.copy()
        
        # Basic returns
        data['returns'] = data['close'].pct_change()
        data['log_returns'] = np.log(data['close'] / data['close'].shift(1))
        
        # Moving averages
        for window in [5, 10, 20, 50]:
            data[f'ma_{window}'] = data['close'].rolling(window=window, min_periods=1).mean()
            data[f'ma_ratio_{window}'] = data['close'] / data[f'ma_{window}']
        
        # Volatility
        data['volatility_10'] = data['returns'].rolling(window=10, min_periods=1).std()
        
        # RSI (Relative Strength Index)
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
        rs = gain / loss.replace(0, np.nan).fillna(1)
        data['rsi_14'] = 100 - (100 / (1 + rs))
        
        # Target variable: future returns (next day's return)
        data['target'] = data['returns'].shift(-1)
        
        # Fill NaN values
        data = data.fillna(0)
        
        # Define features to use
        self.features = [
            'returns', 'log_returns', 
            'ma_ratio_5', 'ma_ratio_10', 'ma_ratio_20', 'ma_ratio_50',
            'volatility_10', 'rsi_14'
        ]
        
        return data
    
    def train(self, df, test_size=0.2):
        """Train the model on historical data."""
        logger.info("Training price prediction model...")
        
        try:
            # Prepare data
            data = self._prepare_features(df)
            
            # Make sure we have enough data
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
            batch_size = min(self.batch_size, len(X_train))
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            
            # Initialize model
            self.model = PriceModel(
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
                
                # Print progress every 20 epochs
                if (epoch + 1) % 20 == 0:
                    logger.info(f"Epoch [{epoch+1}/{self.epochs}], Loss: {total_loss/len(train_loader):.6f}")
            
            # Evaluate on test set
            self.model.eval()
            with torch.no_grad():
                test_pred = self.model(X_test_tensor)
                test_loss = criterion(test_pred, y_test_tensor).item()
            
            logger.info(f"Model trained. Test MSE: {test_loss:.6f}")
            
            return {
                'test_loss': test_loss,
                'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
        except Exception as e:
            logger.error(f"Error during model training: {str(e)}")
            return None
    
    def predict(self, df):
        """Predict future returns using the trained model."""
        if self.model is None:
            logger.error("Model not trained yet.")
            return None
        
        try:
            # Prepare data
            data = self._prepare_features(df)
            
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
            
            self.scaler = metadata['scaler']
            self.features = metadata['features']
            self.hidden_dim = metadata['hidden_dim']
            self.num_layers = metadata['num_layers']
            self.dropout = metadata['dropout']
            
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