# main.py - Unified entry point for AI Trading Bot
import argparse
import logging
import os
from datetime import datetime

# Import config
from config import SYMBOL, TIMEFRAME, INITIAL_CAPITAL, COMMISSION, RISK_PER_TRADE

# Import mode-specific functionality
from run_bot import run_backtest, run_live_trading
from run_with_simulated_data import run_backtest_with_simulated_data
from run_with_recent import run_backtest_with_recent_data
from run_with_yfinance import run_backtest_with_yfinance

# Configure logging
def setup_logging(log_level=logging.INFO):
    """Set up logging configuration with file and console output."""
    os.makedirs('logs', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f"logs/trading_bot_{timestamp}.log"
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_filename}")
    return logger

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='AI Trading Bot with PyTorch',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Main mode selection
    parser.add_argument('--mode', 
                      choices=['backtest', 'paper', 'live', 'simulated', 'recent', 'yfinance'],
                      default='backtest',
                      help='Trading mode to run')
    
    # Strategy selection
    parser.add_argument('--strategy', 
                      choices=['sma', 'ml', 'torch_ml'],
                      default='torch_ml',
                      help='Trading strategy to use')
    
    # Data parameters
    parser.add_argument('--symbol', default=SYMBOL,
                      help='Trading symbol/pair (e.g., BTCUSDT for CCXT, BTC-USD for yfinance)')
    parser.add_argument('--timeframe', default=TIMEFRAME,
                      help='Candlestick timeframe (e.g., 1h, 1d)')
    parser.add_argument('--start-date', help='Start date for backtest (YYYY-MM-DD)')
    parser.add_argument('--end-date', help='End date for backtest (YYYY-MM-DD)')
    
    # Simulation parameters
    parser.add_argument('--days', type=int, default=180,
                      help='Number of days to use for simulated or recent data')
    parser.add_argument('--volatility', type=float, default=0.01,
                      help='Volatility parameter for simulated data')
    
    # Risk parameters
    parser.add_argument('--initial-capital', type=float, default=INITIAL_CAPITAL,
                      help='Initial capital for backtest')
    parser.add_argument('--commission', type=float, default=COMMISSION,
                      help='Commission rate for trades')
    parser.add_argument('--risk-per-trade', type=float, default=RISK_PER_TRADE,
                      help='Maximum risk per trade as a percentage')
    
    # Model parameters
    parser.add_argument('--hidden-dim', type=int, default=64,
                      help='Hidden dimension for neural network models')
    parser.add_argument('--num-layers', type=int, default=2,
                      help='Number of hidden layers for neural network models')
    parser.add_argument('--dropout', type=float, default=0.2,
                      help='Dropout rate for neural network models')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                      help='Learning rate for neural network models')
    parser.add_argument('--epochs', type=int, default=100,
                      help='Number of training epochs for neural network models')
    parser.add_argument('--batch-size', type=int, default=32,
                      help='Batch size for neural network models')
    
    # Advanced options
    parser.add_argument('--log-level', 
                      choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                      default='INFO',
                      help='Logging level')
    parser.add_argument('--save-results', action='store_true',
                      help='Save backtest results to CSV')
    parser.add_argument('--no-plot', action='store_true',
                      help='Disable result plotting')
    
    return parser.parse_args()

def main():
    """Main entry point for the trading bot."""
    # Parse command-line arguments
    args = parse_arguments()
    
    # Set up logging
    log_level = getattr(logging, args.log_level)
    logger = setup_logging(log_level)
    
    logger.info(f"Starting AI Trading Bot in {args.mode} mode with {args.strategy} strategy")
    
    try:
        # Route to appropriate function based on mode
        if args.mode == 'backtest':
            # Regular backtest with exchange data
            run_backtest(
                strategy_name=args.strategy,
                symbol=args.symbol,
                timeframe=args.timeframe,
                start_date=args.start_date,
                end_date=args.end_date
            )
            
        elif args.mode in ['paper', 'live']:
            # Live or paper trading
            paper_trading = args.mode == 'paper'
            
            # Confirmation for live trading
            if not paper_trading:
                confirm = input("Are you sure you want to run LIVE trading? This will use real funds. (yes/no): ")
                if confirm.lower() != 'yes':
                    logger.info("Live trading cancelled.")
                    return
            
            run_live_trading(
                strategy_name=args.strategy,
                symbol=args.symbol,
                timeframe=args.timeframe,
                paper_trading=paper_trading
            )
            
        elif args.mode == 'simulated':
            # Backtest with simulated data
            run_backtest_with_simulated_data(
                strategy_name=args.strategy,
                days=args.days,
                volatility=args.volatility
            )
            
        elif args.mode == 'recent':
            # Backtest with recent data
            run_backtest_with_recent_data(
                strategy_name=args.strategy,
                symbol=args.symbol,
                timeframe=args.timeframe,
                days_back=args.days
            )
            
        elif args.mode == 'yfinance':
            # Backtest with Yahoo Finance data
            run_backtest_with_yfinance(
                strategy_name=args.strategy,
                symbol=args.symbol,
                interval=args.timeframe,
                start_date=args.start_date or '2023-01-01',
                end_date=args.end_date or '2023-12-31'
            )
    
    except KeyboardInterrupt:
        logger.info("Bot stopped by user.")
    except Exception as e:
        logger.error(f"Error running bot: {e}", exc_info=True)

if __name__ == "__main__":
    main()