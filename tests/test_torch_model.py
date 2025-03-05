# tests/test_torch_model.py - Tests for PyTorch price prediction model
import unittest
import pandas as pd
import numpy as np
import torch
import os
import sys
from datetime import datetime, timedelta

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.torch_price_predictor import TorchPricePredictor, PricePredictionModel
from utils.data_provider import SimulatedDataProvider

class TestPricePredictionModel(unittest.TestCase):
    """Tests for PricePredictionModel."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.input_dim = 10
        self.hidden_dim = 32
        self.num_layers = 2
        self.dropout = 0.1
        self.model = PricePredictionModel(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout
        )
    
    def test_model_initialization(self):
        """Test model initialization."""
        # Check model structure
        self.assertIsInstance(self.model, PricePredictionModel)
        self.assertIsInstance(self.model.model, torch.nn.Sequential)
        
        # Number of layers should be 3*num_layers + 1 (each hidden layer has Linear, ReLU, Dropout)
        expected_layers = 3 * self.num_layers + 1
        self.assertEqual(len(list(self.model.model.children())), expected_layers)
    
    def test_forward_pass(self):
        """Test forward pass through the model."""
        # Create random input tensor
        batch_size = 16
        x = torch.randn(batch_size, self.input_dim)
        
        # Forward pass
        output = self.model(x)
        
        # Check output shape
        self.assertEqual(output.shape, (batch_size, 1))


class TestTorchPricePredictor(unittest.TestCase):
    """Tests for TorchPricePredictor."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create predictor
        self.predictor = TorchPricePredictor(
            hidden_dim=32,
            num_layers=2,
            dropout=0.1,
            learning_rate=0.001,
            batch_size=8,
            epochs=5  # Use fewer epochs for testing
        )
        
        # Generate sample data using SimulatedDataProvider
        data_provider = SimulatedDataProvider(
            symbol='TEST/USD',
            timeframe='1d',
            volatility=0.01,
            drift=0.0001
        )
        self.sample_data = data_provider.get_historical_data(limit=200)
    
    def test_create_features(self):
        """Test feature creation."""
        # Create features
        df_features = self.predictor._create_features(self.sample_data)
        
        # Check results
        self.assertIsInstance(df_features, pd.DataFrame)
        self.assertGreater(len(df_features.columns), len(self.sample_data.columns))
        self.assertTrue('returns' in df_features.columns)
        self.assertTrue('target' in df_features.columns)
        
        # Check for NaN values
        self.assertFalse(df_features.isnull().any().any())
    
    def test_prepare_data(self):
        """Test data preparation."""
        # Prepare data
        df_prepared = self.predictor.prepare_data(self.sample_data)
        
        # Check results
        self.assertIsInstance(df_prepared, pd.DataFrame)
        self.assertIsNotNone(self.predictor.features)
        self.assertGreater(len(self.predictor.features), 0)
    
    def test_train_model(self):
        """Test model training."""
        # Train model
        results = self.predictor.train(self.sample_data, test_size=0.2)
        
        # Check results
        self.assertIsNotNone(results)
        self.assertIsNotNone(self.predictor.model)
        self.assertIn('train_loss', results)
        self.assertIn('test_loss', results)
        self.assertIn('r2_train', results)
        self.assertIn('r2_test', results)
        self.assertIn('feature_importance', results)
        
        # Check feature importance
        feature_importance = results['feature_importance']
        self.assertEqual(len(feature_importance), len(self.predictor.features))
    
    def test_predict(self):
        """Test prediction."""
        # Train model first
        self.predictor.train(self.sample_data, test_size=0.2)
        
        # Make predictions
        predictions = self.predictor.predict(self.sample_data)
        
        # Check results
        self.assertIsInstance(predictions, pd.DataFrame)
        self.assertIn('predicted_return', predictions.columns)
        self.assertIn('predicted_price', predictions.columns)
        self.assertEqual(len(predictions), len(self.sample_data))
    
    def test_save_and_load_model(self):
        """Test saving and loading model."""
        # Generate test directory path
        test_dir = 'test_models'
        os.makedirs(test_dir, exist_ok=True)
        model_path = os.path.join(test_dir, 'test_model.pth')
        
        # Train and save model
        self.predictor.train(self.sample_data, test_size=0.2)
        save_success = self.predictor.save_model(model_path)
        self.assertTrue(save_success)
        self.assertTrue(os.path.exists(model_path))
        
        # Create new predictor and load model
        new_predictor = TorchPricePredictor()
        load_success = new_predictor.load_model(model_path)
        self.assertTrue(load_success)
        
        # Check if model was loaded correctly
        self.assertIsNotNone(new_predictor.model)
        self.assertEqual(len(new_predictor.features), len(self.predictor.features))
        
        # Clean up
        if os.path.exists(model_path):
            os.remove(model_path)
        metadata_path = model_path.replace('.pth', '_metadata.joblib')
        if os.path.exists(metadata_path):
            os.remove(metadata_path)
        if os.path.exists(test_dir):
            os.rmdir(test_dir)


if __name__ == '__main__':
    unittest.main()