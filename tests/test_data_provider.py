# tests/test_data_provider.py - Tests for data providers
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.data_provider import (
    SimulatedDataProvider, 
    YFinanceDataProvider, 
    CCXTDataProvider,
    create_data_provider
)

class TestSimulatedDataProvider(unittest.TestCase):
    """Tests for SimulatedDataProvider."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.provider = SimulatedDataProvider(
            symbol='TEST/USD',
            timeframe='1h',
            volatility=0.01,
            drift=0.0001,
            start_price=1000.0
        )
    
    def test_get_historical_data(self):
        """Test getting historical data."""
        # Get 100 hours of data
        df = self.provider.get_historical_data(limit=100)
        
        # Check DataFrame properties
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 100)
        self.assertTrue(all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume']))
        
        # Check data integrity
        self.assertTrue(all(df['high'] >= df['close']))
        self.assertTrue(all(df['high'] >= df['open']))
        self.assertTrue(all(df['low'] <= df['close']))
        self.assertTrue(all(df['low'] <= df['open']))
        self.assertTrue(all(df['volume'] > 0))
    
    def test_get_historical_data_with_dates(self):
        """Test getting historical data with date range."""
        # Define date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        # Get data
        df = self.provider.get_historical_data(
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d')
        )
        
        # Check DataFrame properties
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 7 * 24 - 2)  # Allow for some time difference
        self.assertLess(len(df), 7 * 24 + 2)     # Allow for some time difference
        
        # Check date range
        self.assertGreaterEqual(df.index.min(), pd.Timestamp(start_date))
        self.assertLessEqual(df.index.max(), pd.Timestamp(end_date))
    
    def test_get_current_price(self):
        """Test getting current price."""
        # Generate some data first
        self.provider.get_historical_data(limit=10)
        
        # Get current price
        price = self.provider.get_current_price()
        
        # Check price
        self.assertIsInstance(price, float)
        self.assertGreater(price, 0)


class TestYFinanceDataProvider(unittest.TestCase):
    """Tests for YFinanceDataProvider."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.provider = YFinanceDataProvider(
            symbol='BTC-USD',
            timeframe='1d'
        )
    
    def test_provider_initialization(self):
        """Test provider initialization."""
        self.assertEqual(self.provider.symbol, 'BTC-USD')
        self.assertEqual(self.provider.timeframe, '1d')
        self.assertEqual(self.provider.yf_interval, '1d')
    
    def test_initialization_with_unsupported_timeframe(self):
        """Test initialization with unsupported timeframe."""
        provider = YFinanceDataProvider(
            symbol='BTC-USD',
            timeframe='2h'  # Unsupported
        )
        self.assertEqual(provider.yf_interval, '1h')  # Should default to 1h


class TestDataProviderFactory(unittest.TestCase):
    """Tests for data provider factory function."""
    
    def test_create_simulated_provider(self):
        """Test creating simulated data provider."""
        provider = create_data_provider(
            data_source='simulated',
            symbol='TEST/USD',
            timeframe='1h',
            volatility=0.02,
            drift=0.0002,
            start_price=500.0
        )
        
        self.assertIsInstance(provider, SimulatedDataProvider)
        self.assertEqual(provider.symbol, 'TEST/USD')
        self.assertEqual(provider.timeframe, '1h')
        self.assertEqual(provider.volatility, 0.02)
        self.assertEqual(provider.drift, 0.0002)
        self.assertEqual(provider.start_price, 500.0)
    
    def test_create_yfinance_provider(self):
        """Test creating Yahoo Finance data provider."""
        provider = create_data_provider(
            data_source='yfinance',
            symbol='BTC-USD',
            timeframe='1h'
        )
        
        self.assertIsInstance(provider, YFinanceDataProvider)
        self.assertEqual(provider.symbol, 'BTC-USD')
        self.assertEqual(provider.timeframe, '1h')
    
    def test_create_ccxt_provider(self):
        """Test creating CCXT data provider."""
        provider = create_data_provider(
            data_source='ccxt',
            symbol='BTC/USDT',
            timeframe='1h',
            exchange_id='binance'
        )
        
        self.assertIsInstance(provider, CCXTDataProvider)
        self.assertEqual(provider.symbol, 'BTC/USDT')
        self.assertEqual(provider.timeframe, '1h')
        self.assertEqual(provider.exchange_id, 'binance')


if __name__ == '__main__':
    unittest.main()