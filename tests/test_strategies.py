# tests/test_strategies.py - Tests for trading strategies
import unittest
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timedelta

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from strategies.simple_moving_average import SMAStrategy
from strategies.torch_ml_strategy import TorchMLStrategy
from utils.data_provider import SimulatedDataProvider

class TestSMAStrategy(unittest.TestCase):
    """Tests for SMAStrategy."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create strategy
        self.short_window = 10
        self.long_window = 20
        self.strategy = SMAStrategy(
            short_window=self.short_window,
            long_window=self.long_window
        )
        
        # Generate sample data
        data_provider = SimulatedDataProvider(
            symbol='TEST/USD',
            timeframe='1d',
            volatility=0.02,
            drift=0.0005
        )
        self.sample_data = data_provider.get_historical_data(limit=100)
    
    def test_generate_signals(self):
        """Test signal generation."""
        # Generate signals
        df_signals = self.strategy.generate_signals(self.sample_data)
        
        # Check results
        self.assertIsInstance(df_signals, pd.DataFrame)
        self.assertIn('short_ma', df_signals.columns)
        self.assertIn('long_ma', df_signals.columns)
        self.assertIn('signal', df_signals.columns)
        self.assertIn('position', df_signals.columns)
        
        # Verify signal logic
        df_signals['should_buy'] = (df_signals['short_ma'] > df_signals['long_ma']).astype(int)
        df_signals['should_sell'] = (df_signals['short_ma'] < df_signals['long_ma']).astype(int) * -1
        df_signals['expected_signal'] = df_signals['should_buy'] + df_signals['should_sell']
        
        pd.testing.assert_series_equal(df_signals['signal'], df_signals['expected_signal'])
        
        # Verify that some signals are generated
        self.assertTrue((df_signals['position'] == 2).any() or (df_signals['position'] == -2).any(),
                       "No buy or sell signals were generated")


class TestTorchMLStrategy(unittest.TestCase):
    """Tests for TorchMLStrategy."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create strategy
        self.threshold = 0.005
        self.strategy = TorchMLStrategy(
            threshold=self.threshold,
            retrain_period=30,
            hidden_dim=32,
            num_layers=2,
            dropout=0.1
        )
        
        # Generate sample data
        data_provider = SimulatedDataProvider(
            symbol='TEST/USD',
            timeframe='1d',
            volatility=0.02,
            drift=0.0005
        )
        self.sample_data = data_provider.get_historical_data(limit=200)
    
    def test_train_model(self):
        """Test model training."""
        # Train model
        self.strategy.train_model(self.sample_data)
        
        # Check that model was trained
        self.assertIsNotNone(self.strategy.predictor.model)
        self.assertIsNotNone(self.strategy.last_trained)
    
    def test_generate_signals(self):
        """Test signal generation."""
        # Generate signals
        df_signals = self.strategy.generate_signals(self.sample_data)
        
        # Check results
        self.assertIsInstance(df_signals, pd.DataFrame)
        self.assertIn('predicted_return', df_signals.columns)
        self.assertIn('signal', df_signals.columns)
        self.assertIn('position', df_signals.columns)
        
        # Verify signal logic
        for idx, row in df_signals.iterrows():
            if row['predicted_return'] > self.threshold:
                self.assertEqual(row['signal'], 1, f"Signal should be 1 at {idx} with predicted_return {row['predicted_return']}")
            elif row['predicted_return'] < -self.threshold:
                self.assertEqual(row['signal'], -1, f"Signal should be -1 at {idx} with predicted_return {row['predicted_return']}")
            else:
                self.assertEqual(row['signal'], 0, f"Signal should be 0 at {idx} with predicted_return {row['predicted_return']}")
        
        # Verify that some signals are generated
        self.assertTrue((df_signals['position'] != 0).any(),
                       "No buy or sell signals were generated")


if __name__ == '__main__':
    unittest.main()