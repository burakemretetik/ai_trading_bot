# run_with_yfinance.py - Run backtest with data from Yahoo Finance
import argparse
import pandas as pd
import logging
from datetime import datetime
import os

from config import INITIAL_CAPITAL, COMMISSION
from strategies.simple_moving_average import SMAStrategy
from strategies.torch_ml_strategy import TorchMLStrategy
from utils.backtester import Backtester
from download_data_yfinance import download_crypto_data

# Configure logging
os.makedirs('logs', exist_ok=True)
log_filename = f"logs/yfinance_bot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def run_backtest_with_yfinance(strategy_name, symbol, interval, start_date, end_date):
    """
    Run a backtest using data from Yahoo Finance.
    
    Args:
        strategy_name (str): Name of the strategy to use ('sma' or 'torch_ml')
        symbol (str): Yahoo Finance symbol (e.g., 'BTC-USD', 'ETH-USD')
        interval (str): Data interval ('1h', '1d', etc.)
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format
    """
    logger.info(f"Running backtest for {strategy_name} strategy on {symbol} ({interval}) from {start_date} to {end_date}")
    
    # Download data if needed or load from file
    data_dir = 'data'
    formatted_symbol = symbol.replace('/', '_').replace('-', '_')
    data_file = f"{formatted_symbol}_{interval}.csv"
    data_path = os.path.join(data_dir, data_file)
    
    if not os.path.exists(data_path):
        logger.info(f"Data file not found. Downloading from Yahoo Finance...")
        data_path = download_crypto_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            output_dir=data_dir,
            interval=interval
        )
        
        if not data_path:
            logger.error("Failed to download data. Exiting.")
            return
    
    # Load data
    logger.info(f"Loading data from {data_path}")
    data = pd.read_csv(data_path, index_col=0, parse_dates=True)
    
    if data.empty:
        logger.error("Empty dataset. Check data file.")
        return
    
    logger.info(f"Loaded {len(data)} data points from {data.index.min()} to {data.index.max()}")
    
    # Apply date range filter if specified
    if start_date:
        start_date = pd.to_datetime(start_date)
        data = data[data.index >= start_date]
    
    if end_date:
        end_date = pd.to_datetime(end_date)
        data = data[data.index <= end_date]
    
    logger.info(f"After filtering: {len(data)} data points from {data.index.min()} to {data.index.max()}")
    
    # Initialize strategy
    if strategy_name.lower() == 'sma':
        strategy = SMAStrategy(short_window=20, long_window=50)
    elif strategy_name.lower() == 'torch_ml':
        strategy = TorchMLStrategy(threshold=0.002)
    else:
        logger.error(f"Unknown strategy: {strategy_name}")
        return
    
    # Generate trading signals
    signals = strategy.generate_signals(data)
    
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
    parser = argparse.ArgumentParser(description='Run trading bot with Yahoo Finance data')
    parser.add_argument('--strategy', choices=['sma', 'torch_ml'], default='torch_ml',
                        help='Trading strategy to use: Simple Moving Average (sma) or PyTorch ML (torch_ml)')
    parser.add_argument('--symbol', default='BTC-USD',
                        help='Yahoo Finance symbol (e.g., BTC-USD, ETH-USD)')
    parser.add_argument('--interval', default='1h', choices=['1h', '4h', '1d', '1w'],
                        help='Data interval')
    parser.add_argument('--start-date', default='2023-06-01', 
                        help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', default='2023-12-31',
                        help='End date (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    run_backtest_with_yfinance(
        strategy_name=args.strategy,
        symbol=args.symbol,
        interval=args.interval,
        start_date=args.start_date,
        end_date=args.end_date
    )