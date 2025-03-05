# run_with_recent.py - Run backtest using recent data
import argparse
import pandas as pd
import logging
from datetime import datetime, timedelta
import os

from config import *
from utils.data_loader import DataLoader
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

def run_backtest_with_recent_data(strategy_name, symbol, timeframe, days_back=60):
    """
    Run a backtest using recent data.
    
    Args:
        strategy_name (str): Name of the strategy to use ('sma' or 'torch_ml')
        symbol (str): Trading pair symbol
        timeframe (str): Candlestick timeframe
        days_back (int): Number of days of data to use
    """
    logger.info(f"Running backtest for {strategy_name} strategy on {symbol} ({timeframe}) using {days_back} days of recent data")
    
    # Initialize data loader
    data_loader = DataLoader(symbol=symbol, timeframe=timeframe)
    
    # Load historical data with a higher limit to ensure we get enough data
    limit = days_back * 24 if timeframe == '1h' else days_back  # Adjust based on timeframe
    historical_data = data_loader.get_historical_data(limit=limit)
    
    if historical_data.empty:
        logger.error("No data available. Check API access and symbol validity.")
        return
    
    logger.info(f"Retrieved {len(historical_data)} data points from {historical_data.index.min()} to {historical_data.index.max()}")
    
    # Initialize strategy
    if strategy_name.lower() == 'sma':
        strategy = SMAStrategy(short_window=20, long_window=50)
    elif strategy_name.lower() == 'torch_ml':
        strategy = TorchMLStrategy(threshold=0.002)
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
    parser = argparse.ArgumentParser(description='Run trading bot with recent data')
    parser.add_argument('--strategy', choices=['sma', 'torch_ml'], default='torch_ml',
                        help='Trading strategy to use: Simple Moving Average (sma) or PyTorch ML (torch_ml)')
    parser.add_argument('--symbol', default=SYMBOL,
                        help='Trading symbol/pair')
    parser.add_argument('--timeframe', default=TIMEFRAME,
                        help='Candlestick timeframe')
    parser.add_argument('--days', type=int, default=30, 
                        help='Number of days of recent data to use')
    
    args = parser.parse_args()
    
    run_backtest_with_recent_data(
        strategy_name=args.strategy,
        symbol=args.symbol,
        timeframe=args.timeframe,
        days_back=args.days
    )