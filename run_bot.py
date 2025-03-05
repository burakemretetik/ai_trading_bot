# run_bot.py - Script to run the trading bot with PyTorch support
import argparse
import pandas as pd
import logging
from datetime import datetime
import os

from config import *
from utils.data_loader import DataLoader
from strategies.simple_moving_average import SMAStrategy
from strategies.ml_strategy import MLStrategy
from strategies.torch_ml_strategy import TorchMLStrategy
from utils.backtester import Backtester

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

# Configure logging
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

class RiskManager:
    """Class for managing trading risk."""
    
    def __init__(self, max_risk_per_trade=0.02, max_open_trades=3):
        """
        Initialize risk manager.
        
        Args:
            max_risk_per_trade (float): Maximum percentage of capital to risk per trade
            max_open_trades (int): Maximum number of concurrent open trades
        """
        self.max_risk_per_trade = max_risk_per_trade
        self.max_open_trades = max_open_trades
        self.open_trades = 0
    
    def calculate_position_size(self, capital, entry_price, stop_loss_price):
        """
        Calculate position size based on risk parameters.
        
        Args:
            capital (float): Available capital
            entry_price (float): Entry price
            stop_loss_price (float): Stop loss price
            
        Returns:
            float: Position size in base currency
        """
        # Calculate risk amount
        risk_amount = capital * self.max_risk_per_trade
        
        # Calculate risk per unit
        price_risk = abs(entry_price - stop_loss_price)
        risk_percent = price_risk / entry_price
        
        # Calculate position size
        position_size = risk_amount / price_risk
        
        logger.info(f"Capital: ${capital:.2f}, Risk amount: ${risk_amount:.2f}")
        logger.info(f"Entry: ${entry_price:.2f}, Stop loss: ${stop_loss_price:.2f}, Risk: {risk_percent:.2%}")
        logger.info(f"Position size: {position_size:.6f} units (${position_size * entry_price:.2f})")
        
        return position_size
    
    def can_open_trade(self):
        """Check if a new trade can be opened based on risk limits."""
        return self.open_trades < self.max_open_trades
    
    def open_trade(self):
        """Record a new open trade."""
        if self.can_open_trade():
            self.open_trades += 1
            return True
        return False
    
    def close_trade(self):
        """Record a closed trade."""
        if self.open_trades > 0:
            self.open_trades -= 1
            return True
        return False

def run_backtest(strategy_name, symbol, timeframe, start_date=None, end_date=None):
    """
    Run a backtest for a specified strategy.
    
    Args:
        strategy_name (str): Name of the strategy to use ('sma', 'ml', or 'torch_ml')
        symbol (str): Trading pair symbol
        timeframe (str): Candlestick timeframe
        start_date (str, optional): Start date for backtest (YYYY-MM-DD)
        end_date (str, optional): End date for backtest (YYYY-MM-DD)
    """
    logger.info(f"Running backtest for {strategy_name} strategy on {symbol} ({timeframe})")
    
    # Initialize data loader
    data_loader = DataLoader(symbol=symbol, timeframe=timeframe)
    
    # Load historical data
    historical_data = data_loader.get_historical_data(limit=1000)
    
    # Filter data by date range if specified
    if start_date:
        historical_data = historical_data[historical_data.index >= start_date]
    if end_date:
        historical_data = historical_data[historical_data.index <= end_date]
    
    # Initialize strategy
    if strategy_name.lower() == 'sma':
        strategy = SMAStrategy(short_window=20, long_window=50)
    elif strategy_name.lower() == 'ml':
        strategy = MLStrategy(threshold=0.002)
    elif strategy_name.lower() == 'torch_ml':
        strategy = TorchMLStrategy(threshold=0.002, hidden_dim=64, num_layers=2, dropout=0.2)
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

def run_live_trading(strategy_name, symbol, timeframe, paper_trading=True):
    """
    Run live trading with specified strategy.
    
    Args:
        strategy_name (str): Name of the strategy to use ('sma', 'ml', or 'torch_ml')
        symbol (str): Trading pair symbol
        timeframe (str): Candlestick timeframe
        paper_trading (bool): Whether to use paper trading (simulation)
    """
    logger.info(f"Starting {'paper' if paper_trading else 'live'} trading with {strategy_name} strategy on {symbol} ({timeframe})")
    
    # Initialize data loader
    data_loader = DataLoader(symbol=symbol, timeframe=timeframe)
    
    # Initialize risk manager
    risk_manager = RiskManager(max_risk_per_trade=RISK_PER_TRADE)
    
    # Initialize strategy
    if strategy_name.lower() == 'sma':
        strategy = SMAStrategy(short_window=20, long_window=50)
    elif strategy_name.lower() == 'ml':
        strategy = MLStrategy(threshold=0.002)
    elif strategy_name.lower() == 'torch_ml':
        strategy = TorchMLStrategy(threshold=0.002, hidden_dim=64, num_layers=2, dropout=0.2)
    else:
        logger.error(f"Unknown strategy: {strategy_name}")
        return
    
    # Main trading loop
    try:
        while True:
            # Get latest market data
            market_data = data_loader.get_historical_data(limit=100)
            
            # Generate signals
            signals = strategy.generate_signals(market_data)
            
            # Get the latest signal
            latest_signal = signals.iloc[-1]
            
            # Check for trade signals
            if latest_signal['position'] == 2 and risk_manager.can_open_trade():  # Buy signal
                # Calculate position size
                current_price = latest_signal['close']
                stop_loss_price = current_price * 0.95  # 5% stop loss
                
                position_size = risk_manager.calculate_position_size(
                    capital=INITIAL_CAPITAL,  # Use actual account balance in production
                    entry_price=current_price,
                    stop_loss_price=stop_loss_price
                )
                
                # Execute buy order (simulated in paper trading)
                logger.info(f"BUY SIGNAL: {symbol} at ${current_price:.2f}")
                logger.info(f"Position size: {position_size:.6f} {symbol.split('/')[0]}")
                logger.info(f"Stop loss: ${stop_loss_price:.2f}")
                
                if not paper_trading:
                    # Place actual order through exchange API
                    # TODO: Implement actual order placement
                    pass
                
                # Record open trade
                risk_manager.open_trade()
                
            elif latest_signal['position'] == -2:  # Sell signal
                # Execute sell order (simulated in paper trading)
                logger.info(f"SELL SIGNAL: {symbol} at ${latest_signal['close']:.2f}")
                
                if not paper_trading:
                    # Place actual order through exchange API
                    # TODO: Implement actual order placement
                    pass
                
                # Record closed trade
                risk_manager.close_trade()
            
            # Wait for the next candle
            # In production, you would schedule this to run at specific intervals
            import time
            time.sleep(60)  # Wait for 60 seconds in this example
            
    except KeyboardInterrupt:
        logger.info("Trading stopped by user.")
    except Exception as e:
        logger.error(f"Error in trading loop: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run trading bot')
    parser.add_argument('--mode', choices=['backtest', 'paper', 'live'], default='backtest',
                        help='Trading mode: backtest, paper trading, or live trading')
    parser.add_argument('--strategy', choices=['sma', 'ml', 'torch_ml'], default='sma',
                        help='Trading strategy to use: Simple Moving Average (sma), scikit-learn ML (ml), or PyTorch ML (torch_ml)')
    parser.add_argument('--symbol', default=SYMBOL,
                        help='Trading symbol/pair')
    parser.add_argument('--timeframe', default=TIMEFRAME,
                        help='Candlestick timeframe')
    parser.add_argument('--start-date', help='Start date for backtest (YYYY-MM-DD)')
    parser.add_argument('--end-date', help='End date for backtest (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    logger.info(f"Starting trading bot in {args.mode} mode with {args.strategy} strategy")
    
    if args.mode == 'backtest':
        run_backtest(
            strategy_name=args.strategy,
            symbol=args.symbol,
            timeframe=args.timeframe,
            start_date=args.start_date,
            end_date=args.end_date
        )
    elif args.mode == 'paper':
        run_live_trading(
            strategy_name=args.strategy,
            symbol=args.symbol,
            timeframe=args.timeframe,
            paper_trading=True
        )
    elif args.mode == 'live':
        # Confirmation for live trading
        confirm = input("Are you sure you want to run LIVE trading? This will use real funds. (yes/no): ")
        if confirm.lower() == 'yes':
            run_live_trading(
                strategy_name=args.strategy,
                symbol=args.symbol,
                timeframe=args.timeframe,
                paper_trading=False
            )
        else:
            logger.info("Live trading cancelled.")