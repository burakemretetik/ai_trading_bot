# tests/test_basic.py - Basic unit tests for trading bot
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys
import tempfile

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import modules to test
from utils.data import DataProvider
from strategies.sma_strategy import SMAStrategy
from strategies.ml_strategy import MLStrategy
from utils.backtester import Backtester

# Test data generator
def generate_test_data(days=100):
    """Generate test price data."""
    np.random.seed(42)  # For reproducibility
    
    # Generate dates
    end_date = datetime.now()
    dates = [end_date - timedelta(days=i) for i in range(days)]
    dates.reverse()
    
    # Generate prices with a slight uptrend and some randomness
    base_price = 100.0
    trend = np.linspace(0, 0.2, days)  # Slight uptrend
    noise = np.random.normal(0, 0.02, days)  # Random noise
    returns = trend + noise
    
    prices = base_price * np.cumprod(1 + returns)
    
    # Create test DataFrame
    df = pd.DataFrame({
        'open': prices * (1 - np.random.uniform(0, 0.01, days)),
        'high': prices * (1 + np.random.uniform(0, 0.01, days)),
        'low': prices * (1 - np.random.uniform(0, 0.01, days)),
        'close': prices,
        'volume': np.random.uniform(1000, 10000, days)
    }, index=dates)
    
    return df

# Tests for data provider
class TestDataProvider:
    
    def test_initialization(self):
        """Test data provider initialization."""
        data_provider = DataProvider('BTCUSDT', '1d', 'simulated')
        assert data_provider.symbol == 'BTCUSDT'
        assert data_provider.timeframe == '1d'
        assert data_provider.data_source == 'simulated'
    
    def test_simulated_data(self):
        """Test simulated data generation."""
        data_provider = DataProvider('BTCUSDT', '1d', 'simulated')
        data = data_provider.get_historical_data(limit=50)
        
        # Check if we got the data
        assert not data.empty
        assert len(data) == 50
        
        # Check required columns
        for col in ['open', 'high', 'low', 'close', 'volume']:
            assert col in data.columns
        
        # Check data integrity
        assert (data['high'] >= data['low']).all()
        assert (data['high'] >= data['close']).all()
        assert (data['high'] >= data['open']).all()
        assert (data['low'] <= data['close']).all()
        assert (data['low'] <= data['open']).all()
    
    def test_csv_data(self):
        """Test loading data from CSV."""
        # Generate test data and save to CSV
        test_data = generate_test_data(50)
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            test_data.to_csv(f.name)
            filepath = f.name
        
        try:
            # Load data from CSV
            data_provider = DataProvider('TEST', '1d', 'csv', filepath=filepath)
            data = data_provider.get_historical_data()
            
            # Check if we got the data
            assert not data.empty
            assert len(data) == 50
            
            # Check required columns
            for col in ['open', 'high', 'low', 'close', 'volume']:
                assert col in data.columns
        finally:
            # Clean up
            os.unlink(filepath)

# Tests for SMA strategy
class TestSMAStrategy:
    
    def test_initialization(self):
        """Test SMA strategy initialization."""
        strategy = SMAStrategy(short_window=10, long_window=30)
        assert strategy.short_window == 10
        assert strategy.long_window == 30
    
    def test_window_validation(self):
        """Test SMA strategy window validation."""
        # Should swap windows if short > long
        strategy = SMAStrategy(short_window=50, long_window=20)
        assert strategy.short_window == 20
        assert strategy.long_window == 50
    
    def test_signal_generation(self):
        """Test SMA strategy signal generation."""
        test_data = generate_test_data(100)
        strategy = SMAStrategy(short_window=10, long_window=30)
        
        signals = strategy.generate_signals(test_data)
        
        # Check if we got signals
        assert not signals.empty
        assert 'signal' in signals.columns
        assert 'position' in signals.columns
        
        # Check for buy/sell signals
        assert (signals['position'] != 0).any(), "No trading signals generated"
    
    def test_generate_signals_insufficient_data(self):
        """Test SMA strategy with insufficient data."""
        test_data = generate_test_data(20)  # Too few data points for default windows
        strategy = SMAStrategy(short_window=50, long_window=100)
        
        signals = strategy.generate_signals(test_data)
        
        # Should return data with empty signals
        assert not signals.empty
        assert 'signal' in signals.columns
        assert 'position' in signals.columns
        assert (signals['signal'] == 0).all(), "Signals should all be zero with insufficient data"

# Tests for ML strategy
class TestMLStrategy:
    
    def test_initialization(self):
        """Test ML strategy initialization."""
        strategy = MLStrategy(threshold=0.005, retrain_period=20)
        assert strategy.threshold == 0.005
        assert strategy.retrain_period == 20
    
    def test_validation_with_invalid_data(self):
        """Test ML strategy validation with invalid data."""
        strategy = MLStrategy()
        
        # Empty DataFrame
        empty_df = pd.DataFrame()
        assert strategy._validate_input_data(empty_df) is False
        
        # Missing required columns
        invalid_df = pd.DataFrame({'price': [100, 101, 102]})
        assert strategy._validate_input_data(invalid_df) is False
        
        # Insufficient data points
        short_df = generate_test_data(10)  # Too few data points
        assert strategy._validate_input_data(short_df) is False

# Tests for Backtester
class TestBacktester:
    
    def test_initialization(self):
        """Test backtester initialization."""
        backtester = Backtester(initial_capital=5000, commission=0.002)
        assert backtester.initial_capital == 5000
        assert backtester.commission == 0.002
    
    def test_validation_with_invalid_signals(self):
        """Test backtester validation with invalid signals."""
        backtester = Backtester()
        
        # Empty DataFrame
        empty_df = pd.DataFrame()
        assert backtester._validate_signals(empty_df) is False
        
        # Missing required columns
        invalid_df = pd.DataFrame({'price': [100, 101, 102]})
        assert backtester._validate_signals(invalid_df) is False
    
    def test_backtest_with_signals(self):
        """Test basic backtest with signals."""
        # Generate test data and signals
        test_data = generate_test_data(100)
        strategy = SMAStrategy(short_window=10, long_window=30)
        signals = strategy.generate_signals(test_data)
        
        # Run backtest
        backtester = Backtester(initial_capital=10000, commission=0.001)
        performance = backtester.run(signals)
        
        # Check if backtest produced results
        assert performance is not None
        assert 'final_equity' in performance
        assert 'total_return' in performance
        
        # Check if results dataframe is created
        assert backtester.results is not None
        assert 'total_equity' in backtester.results.columns
        
        # Check if final equity is reasonable (not negative or zero)
        assert performance['final_equity'] > 0