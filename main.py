# main.py - Simplified entry point for trading bot
import argparse
import logging
import os
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

# Import configuration
from config import CONFIG, MODE, STRATEGY, SYMBOL, TIMEFRAME, INITIAL_CAPITAL, COMMISSION

# Import components
from utils.data import get_data_provider
from strategies.sma_strategy import SMAStrategy
from strategies.ml_strategy import MLStrategy
from utils.backtester import Backtester
from utils.visualizer import Visualizer

# Setup logging
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='AI Trading Bot',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Mode selection
    parser.add_argument('--mode', 
                      choices=['backtest', 'paper', 'live', 'simulated'],
                      default=MODE,
                      help='Trading mode to run')
    
    # Strategy selection
    parser.add_argument('--strategy', 
                      choices=['sma', 'ml'],
                      default=STRATEGY,
                      help='Trading strategy to use')
    
    # Data parameters
    parser.add_argument('--symbol', default=SYMBOL,
                      help='Trading symbol/pair (e.g., BTCUSDT)')
    parser.add_argument('--timeframe', default=TIMEFRAME,
                      help='Candlestick timeframe (e.g., 1h, 1d)')
    parser.add_argument('--data-source', default='ccxt',
                      choices=['ccxt', 'yfinance', 'simulated', 'csv'],
                      help='Data source')
    parser.add_argument('--start-date', help='Start date for backtest (YYYY-MM-DD)')
    parser.add_argument('--end-date', help='End date for backtest (YYYY-MM-DD)')
    
    # Risk parameters
    parser.add_argument('--initial-capital', type=float, default=INITIAL_CAPITAL,
                      help='Initial capital for backtest')
    parser.add_argument('--commission', type=float, default=COMMISSION,
                      help='Commission rate for trades')
    
    # Output options
    parser.add_argument('--output-dir', default='results',
                      help='Directory for output files')
    parser.add_argument('--save-results', action='store_true',
                      help='Save backtest results to CSV')
    parser.add_argument('--no-plot', action='store_true',
                      help='Disable result plotting')
    
    return parser.parse_args()

def get_strategy(strategy_name, **kwargs):
    """
    Get strategy instance based on name.
    
    Args:
        strategy_name: Name of the strategy ('sma' or 'ml')
        **kwargs: Additional strategy parameters
        
    Returns:
        Strategy instance
    """
    if strategy_name.lower() == 'sma':
        short_window = kwargs.get('short_window', 20)
        long_window = kwargs.get('long_window', 50)
        return SMAStrategy(short_window=short_window, long_window=long_window)
    elif strategy_name.lower() == 'ml':
        threshold = kwargs.get('threshold', 0.002)
        retrain_period = kwargs.get('retrain_period', 30)
        hidden_dim = kwargs.get('hidden_dim', 64)
        num_layers = kwargs.get('num_layers', 2)
        dropout = kwargs.get('dropout', 0.2)
        return MLStrategy(
            threshold=threshold,
            retrain_period=retrain_period,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )
    else:
        logger.error(f"Unknown strategy: {strategy_name}")
        return None

def run_backtest(args):
    """
    Run backtest with specified parameters.
    
    Args:
        args: Command line arguments
    """
    logger.info(f"Running backtest with {args.strategy} strategy on {args.symbol}")
    
    # Create data provider
    data_provider = get_data_provider(
        symbol=args.symbol,
        timeframe=args.timeframe,
        data_source=args.data_source
    )
    
    # Get historical data
    historical_data = data_provider.get_historical_data(
        limit=1000,
        start_date=args.start_date,
        end_date=args.end_date
    )
    
    if historical_data.empty:
        logger.error("No data available for backtest")
        return
    
    # Get strategy
    strategy_params = CONFIG.get('model', {})
    strategy = get_strategy(args.strategy, **strategy_params)
    
    if strategy is None:
        return
    
    # Generate signals
    signals = strategy.generate_signals(historical_data)
    
    # Run backtest
    backtester = Backtester(
        initial_capital=args.initial_capital,
        commission=args.commission
    )
    performance = backtester.run(signals)
    
    if not performance:
        logger.error("Backtest failed")
        return
    
    # Print performance metrics
    logger.info("\nBacktest Performance:")
    for key, value in performance.items():
        if key in ['total_return', 'annualized_return', 'win_rate', 'max_drawdown']:
            logger.info(f"  {key}: {value:.2f}%")
        elif key in ['initial_equity', 'final_equity']:
            logger.info(f"  {key}: ${value:.2f}")
        elif key in ['sharpe_ratio', 'profit_factor']:
            logger.info(f"  {key}: {value:.4f}")
        else:
            logger.info(f"  {key}: {value}")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save results
    if args.save_results:
        results_file = os.path.join(args.output_dir, f"{args.strategy}_{args.symbol}_{args.timeframe}_results.csv")
        backtester.results.to_csv(results_file)
        logger.info(f"Results saved to {results_file}")
        
        # Save trade summary
        trade_summary = backtester.get_trade_summary()
        if not trade_summary.empty:
            trades_file = os.path.join(args.output_dir, f"{args.strategy}_{args.symbol}_{args.timeframe}_trades.csv")
            trade_summary.to_csv(trades_file)
            logger.info(f"Trade summary saved to {trades_file}")
    
    # Plot results
    if not args.no_plot:
        # Plot price with signals
        signals_plot = os.path.join(args.output_dir, f"{args.strategy}_{args.symbol}_{args.timeframe}_signals.png")
        Visualizer.plot_price_with_signals(signals, signals_plot)
        
        # Plot equity and drawdown
        equity_plot = os.path.join(args.output_dir, f"{args.strategy}_{args.symbol}_{args.timeframe}_equity.png")
        Visualizer.plot_equity_and_drawdown(backtester.results, equity_plot)
        
        # Plot returns distribution
        returns_plot = os.path.join(args.output_dir, f"{args.strategy}_{args.symbol}_{args.timeframe}_returns.png")
        Visualizer.plot_returns_distribution(backtester.results, returns_plot)

def run_multiple_backtests(args):
    """
    Run backtest with multiple strategies for comparison.
    
    Args:
        args: Command line arguments
    """
    logger.info(f"Running comparison backtest on {args.symbol}")
    
    # Create data provider
    data_provider = get_data_provider(
        symbol=args.symbol,
        timeframe=args.timeframe,
        data_source=args.data_source
    )
    
    # Get historical data
    historical_data = data_provider.get_historical_data(
        limit=1000,
        start_date=args.start_date,
        end_date=args.end_date
    )
    
    if historical_data.empty:
        logger.error("No data available for backtest")
        return
    
    # Strategies to compare
    strategies = {
        'SMA(20,50)': SMAStrategy(short_window=20, long_window=50),
        'SMA(10,30)': SMAStrategy(short_window=10, long_window=30),
        'ML': MLStrategy(threshold=0.002)
    }
    
    # Run backtest for each strategy
    results = {}
    performances = {}
    
    for name, strategy in strategies.items():
        logger.info(f"Running backtest for {name} strategy")
        
        # Generate signals
        signals = strategy.generate_signals(historical_data)
        
        # Run backtest
        backtester = Backtester(
            initial_capital=args.initial_capital,
            commission=args.commission
        )
        performance = backtester.run(signals)
        
        if performance:
            results[name] = backtester.results
            performances[name] = performance
    
    # Compare strategies
    if results:
        # Create output directory if it doesn't exist
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Plot strategy comparison
        comparison_plot = os.path.join(args.output_dir, f"strategy_comparison_{args.symbol}_{args.timeframe}.png")
        Visualizer.plot_strategy_comparison(results, comparison_plot)
        
        # Print performance comparison
        logger.info("\nStrategy Performance Comparison:")
        
        # Create comparison table
        comparison = pd.DataFrame(performances).T
        
        # Select key metrics
        metrics = ['total_return', 'annualized_return', 'max_drawdown', 'sharpe_ratio', 'total_trades', 'win_rate']
        comparison = comparison[metrics]
        
        # Format and print comparison
        comparison_str = comparison.to_string(float_format=lambda x: f"{x:.2f}")
        logger.info(f"\n{comparison_str}")
        
        # Save comparison to CSV
        if args.save_results:
            comparison_file = os.path.join(args.output_dir, f"strategy_comparison_{args.symbol}_{args.timeframe}.csv")
            comparison.to_csv(comparison_file)
            logger.info(f"Comparison results saved to {comparison_file}")

def run_paper_trading(args):
    """
    Run paper trading simulation.
    
    Args:
        args: Command line arguments
    """
    logger.info(f"Starting paper trading with {args.strategy} strategy on {args.symbol}")
    
    # Get strategy
    strategy_params = CONFIG.get('model', {})
    strategy = get_strategy(args.strategy, **strategy_params)
    
    if strategy is None:
        return
    
    # Create data provider
    data_provider = get_data_provider(
        symbol=args.symbol,
        timeframe=args.timeframe,
        data_source=args.data_source
    )
    
    # Simple paper trading simulation
    logger.info("Paper trading simulation not fully implemented")
    logger.info("To implement paper trading, you would:")
    logger.info("1. Periodically fetch new data")
    logger.info("2. Generate signals on the latest data")
    logger.info("3. Execute trades based on signals")
    logger.info("4. Track portfolio performance")
    
    # As a demonstration, just fetch recent data and show signals
    recent_data = data_provider.get_historical_data(limit=100)
    
    if recent_data.empty:
        logger.error("No data available")
        return
    
    signals = strategy.generate_signals(recent_data)
    
    # Check if we have a signal in the most recent candle
    latest_signal = signals.iloc[-1]
    
    logger.info(f"Latest price: ${latest_signal['close']:.2f}")
    
    if latest_signal['position'] == 2:  # Buy signal
        logger.info("SIGNAL: BUY")
    elif latest_signal['position'] == -2:  # Sell signal
        logger.info("SIGNAL: SELL")
    else:
        logger.info("SIGNAL: None (Hold current position)")

def confirm_live_trading():
    """Ask for confirmation before live trading."""
    print("\n⚠️  WARNING: LIVE TRADING MODE ⚠️")
    print("This will use real funds and execute actual trades.")
    print("Are you absolutely sure you want to continue?")
    
    confirm = input("Type 'yes' to confirm: ")
    return confirm.lower() == 'yes'

def run_live_trading(args):
    """
    Run live trading with specified parameters.
    
    Args:
        args: Command line arguments
    """
    # Confirm live trading
    if not confirm_live_trading():
        logger.info("Live trading cancelled")
        return
    
    logger.info(f"Starting live trading with {args.strategy} strategy on {args.symbol}")
    logger.info("Live trading not implemented in this simplified version")
    logger.info("To implement live trading, you would:")
    logger.info("1. Connect to exchange API with authentication")
    logger.info("2. Implement order placement logic")
    logger.info("3. Implement risk management")
    logger.info("4. Implement position tracking")
    logger.info("5. Implement error handling and failsafes")

def main():
    """Main entry point for the trading bot."""
    # Parse command-line arguments
    args = parse_arguments()
    
    try:
        # Route to appropriate function based on mode
        if args.mode == 'backtest':
            run_backtest(args)
        elif args.mode == 'compare':
            run_multiple_backtests(args)
        elif args.mode == 'paper':
            run_paper_trading(args)
        elif args.mode == 'live':
            run_live_trading(args)
        else:
            logger.error(f"Unknown mode: {args.mode}")
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Error running bot: {e}", exc_info=True)

if __name__ == "__main__":
    main()