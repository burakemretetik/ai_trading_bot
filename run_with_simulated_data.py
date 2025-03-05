# run_with_simulated_data.py - Run bot with generated data for testing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from datetime import datetime, timedelta
import os

from config import *
from strategies.simple_moving_average import SMAStrategy
from strategies.torch_ml_strategy import TorchMLStrategy
from utils.backtester import Backtester

# Configure logging
os.makedirs('logs', exist_ok=True)
log_filename = f"logs/trading_bot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def generate_simulated_data(days=365, volatility=0.01):
    """Generate simulated price data."""
    logger.info(f"Generating simulated data for {days} days")
    
    # Generate dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date, freq='1h')  # lowercase 'h'
    
    # Generate price series (random walk with drift)
    np.random.seed(42)  # For reproducibility
    returns = np.random.normal(0.0001, volatility, len(dates))
    price = 1000 * np.cumprod(1 + returns)
    
    # Add some patterns for ML to detect
    t = np.arange(len(dates))
    cycles = 0.02 * np.sin(t/24*2*np.pi) + 0.01 * np.sin(t/168*2*np.pi)
    price = price * (1 + cycles)
    
    # Create OHLCV dataframe
    df = pd.DataFrame(index=dates)
    df['close'] = price
    
    # Handle the first element separately to avoid NaN
    df['open'] = df['close'].copy()
    df['open'].iloc[1:] = df['close'].iloc[:-1].values * (1 + np.random.normal(0, 0.001, len(dates)-1))
    
    df['high'] = np.maximum(df['open'], df['close']) * (1 + np.abs(np.random.normal(0, 0.003, len(dates))))
    df['low'] = np.minimum(df['open'], df['close']) * (1 - np.abs(np.random.normal(0, 0.003, len(dates))))
    df['volume'] = np.random.lognormal(10, 1, len(dates)) * (1 + np.abs(returns * 10))
    
    logger.info(f"Generated {len(df)} data points from {df.index[0]} to {df.index[-1]}")
    
    return df

def run_backtest_with_simulated_data(strategy_name='sma'):
    """Run backtest using simulated data."""
    logger.info(f"Running backtest with simulated data using {strategy_name} strategy")
    
    # Generate simulated data
    historical_data = generate_simulated_data(days=180)
    
    # Initialize strategy
    if strategy_name.lower() == 'sma':
        strategy = SMAStrategy(short_window=20, long_window=50)
    elif strategy_name.lower() == 'torch_ml':
        strategy = TorchMLStrategy(threshold=0.001)
    else:
        logger.error(f"Unknown strategy: {strategy_name}")
        return
    
    # Generate trading signals
    signals = strategy.generate_signals(historical_data)
    
    # Run backtest
    backtester = Backtester(initial_capital=INITIAL_CAPITAL, commission=COMMISSION)
    performance = backtester.run(signals)
    
    # Print performance metrics
    logger.info("Backtest Performance:")
    for key, value in performance.items():
        if key in ['total_return', 'win_rate', 'max_drawdown']:
            logger.info(f"  {key}: {value:.2f}%")
        elif key in ['initial_equity', 'final_equity']:
            logger.info(f"  {key}: ${value:.2f}")
        elif key == 'sharpe_ratio':
            logger.info(f"  {key}: {value:.4f}")
        else:
            logger.info(f"  {key}: {value}")
    
    # Plot results
    backtester.plot_results()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run trading bot with simulated data')
    parser.add_argument('--strategy', choices=['sma', 'torch_ml'], default='sma',
                        help='Trading strategy to use: Simple Moving Average (sma) or PyTorch ML (torch_ml)')
    
    args = parser.parse_args()
    
    run_backtest_with_simulated_data(strategy_name=args.strategy)