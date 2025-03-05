# models/price_predictor.py - ML model for price prediction
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib
import logging

logger = logging.getLogger(__name__)

class PricePredictor:
    """Class for predicting price movements using ML."""
    
    def __init__(self, n_estimators=100, max_depth=10):
        """Initialize the price predictor with model parameters."""
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.features = None
    
    def _create_features(self, df):
        """Create technical indicators as features."""
        data = df.copy()
        
        # Price-based features
        data['returns'] = data['close'].pct_change()
        data['log_returns'] = np.log(data['close'] / data['close'].shift(1))
        
        # Moving averages
        for window in [5, 10, 20, 50, 100]:
            data[f'ma_{window}'] = data['close'].rolling(window=window).mean()
            data[f'ma_ratio_{window}'] = data['close'] / data[f'ma_{window}']
        
        # Volatility
        for window in [5, 10, 20]:
            data[f'volatility_{window}'] = data['returns'].rolling(window=window).std()
        
        # RSI (Relative Strength Index)
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['rsi_14'] = 100 - (100 / (1 + rs))
        
        # MACD (Moving Average Convergence Divergence)
        data['ema_12'] = data['close'].ewm(span=12, adjust=False).mean()
        data['ema_26'] = data['close'].ewm(span=26, adjust=False).mean()
        data['macd'] = data['ema_12'] - data['ema_26']
        data['macd_signal'] = data['macd'].ewm(span=9, adjust=False).mean()
        data['macd_hist'] = data['macd'] - data['macd_signal']
        
        # Volume-based features
        data['volume_ma_5'] = data['volume'].rolling(window=5).mean()
        data['volume_ratio'] = data['volume'] / data['volume_ma_5']
        
        # Target variable: future returns (next day's return)
        data['target'] = data['returns'].shift(-1)
        
        # Drop rows with NaN values
        data = data.dropna()
        
        return data
    
    def prepare_data(self, df):
        """Prepare data for training or prediction."""
        # Create features
        data = self._create_features(df)
        
        # Define features to use
        self.features = [col for col in data.columns if col not in [
            'open', 'high', 'low', 'close', 'volume', 'target'
        ]]
        
        return data
    
    def train(self, df, test_size=0.2):
        """Train the model on historical data."""
        logger.info("Training price prediction model...")
        
        # Prepare data
        data = self.prepare_data(df)
        
        # Split features and target
        X = data[self.features]
        y = data['target']
        
        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train the model
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate the model
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        
        logger.info(f"Model trained. R² score (train): {train_score:.4f}, R² score (test): {test_score:.4f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.features,
            'importance': self.model.feature_importances_
        })
        feature_importance = feature_importance.sort_values('importance', ascending=False)
        
        logger.info(f"Top 5 important features: {feature_importance.head(5)['feature'].tolist()}")
        
        return {
            'train_score': train_score,
            'test_score': test_score,
            'feature_importance': feature_importance
        }
    
    def predict(self, df):
        """Predict future returns using the trained model."""
        if self.model is None:
            logger.error("Model not trained yet.")
            return None
        
        # Prepare data
        data = self.prepare_data(df)
        
        # Extract features
        X = data[self.features]
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        predictions = self.model.predict(X_scaled)
        
        # Add predictions to dataframe
        data['predicted_return'] = predictions
        data['predicted_price'] = data['close'] * (1 + data['predicted_return'])
        
        return data[['close', 'predicted_return', 'predicted_price']]
    
    def save_model(self, filepath='models/price_predictor.joblib'):
        """Save the trained model to a file."""
        if self.model is None:
            logger.error("No trained model to save.")
            return False
        
        try:
            joblib.dump({
                'model': self.model,
                'scaler': self.scaler,
                'features': self.features
            }, filepath)
            logger.info(f"Model saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
    
    def load_model(self, filepath='models/price_predictor.joblib'):
        """Load a trained model from a file."""
        try:
            model_data = joblib.load(filepath)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.features = model_data['features']
            logger.info(f"Model loaded from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False


# strategies/ml_strategy.py - ML-based trading strategy
import pandas as pd
import numpy as np
from models.price_predictor import PricePredictor
import logging

logger = logging.getLogger(__name__)

class MLStrategy:
    """Trading strategy based on machine learning predictions."""
    
    def __init__(self, threshold=0.001, retrain_period=30):
        """
        Initialize ML strategy.
        
        Args:
            threshold (float): Minimum predicted return to enter a position
            retrain_period (int): Number of days before retraining the model
        """
        self.predictor = PricePredictor()
        self.threshold = threshold
        self.retrain_period = retrain_period
        self.last_trained = None
    
    def train_model(self, data):
        """Train the ML model on historical data."""
        logger.info("Training ML model for trading strategy...")
        self.predictor.train(data)
        self.last_trained = data.index[-1]
    
    def generate_signals(self, data):
        """Generate buy/sell signals based on ML predictions."""
        logger.info("Generating signals using ML strategy...")
        
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