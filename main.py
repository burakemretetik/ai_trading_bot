# main.py - Enhanced entry point for trading bot
import argparse
import logging
import os
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import traceback
import signal
import sys

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
    """Parse command line arguments with improved options."""
    parser = argparse.ArgumentParser(
        description='AI Trading Bot',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Mode selection
    parser.add_argument('--mode', 
                      choices=['backtest', 'paper', 'live', 'simulated', 'compare', 'optimize'],
                      default=MODE,
                      help='Trading mode to run')
    
    # Strategy selection
    parser.add_argument('--strategy', 
                      choices=['sma', 'ml', 'torch_ml'],
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
    parser.add_argument('--csv-file', help='CSV file path for data (when using csv data source)')
    
    # Risk parameters
    parser.add_argument('--initial-capital', type=float, default=INITIAL_CAPITAL,
                      help='Initial capital for backtest')
    parser.add_argument('--commission', type=float, default=COMMISSION,
                      help='Commission rate for trades')
    parser.add_argument('--risk-per-trade', type=float, default=CONFIG.get('risk_per_trade', 0.02),
                      help='Percentage of capital to risk per trade (e.g., 0.02 for 2%%)')
    parser.add_argument('--slippage', type=float, default=0.0,
                      help='Slippage as percentage of price (e.g., 0.001 for 0.1%%)')
    
    # Strategy parameters
    parser.add_argument('--short-window', type=int, default=20,
                      help='Short moving average window (SMA strategy)')
    parser.add_argument('--long-window', type=int, default=50,
                      help='Long moving average window (SMA strategy)')
    parser.add_argument('--ml-threshold', type=float, default=0.002,
                      help='Prediction threshold for ML strategy')
    parser.add_argument('--retrain-period', type=int, default=30,
                      help='Retrain period for ML strategy (in days)')
    
    # Output options
    parser.add_argument('--output-dir', default='results',
                      help='Directory for output files')
    parser.add_argument('--save-results', action='store_true',
                      help='Save backtest results to CSV')
    parser.add_argument('--no-plot', action='store_true',
                      help='Disable result plotting')
    parser.add_argument('--verbose', action='store_true',
                      help='Enable verbose output')
    
    # Advanced options
    parser.add_argument('--monte-carlo', action='store_true',
                      help='Run Monte Carlo analysis for backtest results')
    parser.add_argument('--mc-iterations', type=int, default=1000,
                      help='Number of Monte Carlo iterations')
    
    return parser.parse_args()

def setup_signal_handler():
    """Set up signal handler for graceful shutdown."""
    def signal_handler(sig, frame):
        logger.info("Shutting down gracefully...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

def get_strategy(strategy_name, **kwargs):
    """
    Get strategy instance based on name with improved error handling.
    
    Args:
        strategy_name: Name of the strategy ('sma', 'ml', or 'torch_ml')
        **kwargs: Additional strategy parameters
        
    Returns:
        Strategy instance or None if error occurs
    """
    try:
        if strategy_name.lower() == 'sma':
            short_window = kwargs.get('short_window', 20)
            long_window = kwargs.get('long_window', 50)
            filter_noise = kwargs.get('filter_noise', True)
            confirm_days = kwargs.get('confirm_days', 1)
            
            return SMAStrategy(
                short_window=short_window, 
                long_window=long_window,
                filter_noise=filter_noise,
                confirm_days=confirm_days
            )
        elif strategy_name.lower() in ['ml', 'torch_ml']:
            threshold = kwargs.get('threshold', 0.002)
            retrain_period = kwargs.get('retrain_period', 30)
            hidden_dim = kwargs.get('hidden_dim', 64)
            num_layers = kwargs.get('num_layers', 2)
            dropout = kwargs.get('dropout', 0.2)
            feature_engineering = kwargs.get('feature_engineering', 'basic')
            stop_loss = kwargs.get('stop_loss', 0.05)
            
            return MLStrategy(
                threshold=threshold,
                retrain_period=retrain_period,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
                feature_engineering=feature_engineering,
                stop_loss=stop_loss
            )
        else:
            logger.error(f"Unknown strategy: {strategy_name}")
            return None
    except Exception as e:
        logger.error(f"Error creating strategy {strategy_name}: {e}")
        return None

def get_data_for_backtest(args):
    """
    Get historical data for backtest with improved error handling.
    
    Args:
        args: Command line arguments
        
    Returns:
        DataFrame with historical data or None if error
    """
    try:
        # Create data provider with appropriate parameters
        data_provider_params = {}
        
        if args.data_source == 'csv' and args.csv_file:
            data_provider_params['filepath'] = args.csv_file
        
        data_provider = get_data_provider(
            symbol=args.symbol,
            timeframe=args.timeframe,
            data_source=args.data_source,
            **data_provider_params
        )
        
        # Get historical data
        historical_data = data_provider.get_historical_data(
            limit=1000,
            start_date=args.start_date,
            end_date=args.end_date
        )
        
        if historical_data.empty:
            logger.error("No data available for backtest")
            return None
        
        return historical_data
    except Exception as e:
        logger.error(f"Error getting historical data: {e}")
        return None

def run_backtest(args):
    """
    Run backtest with specified parameters and improved error handling.
    
    Args:
        args: Command line arguments
    """
    logger.info(f"Running backtest with {args.strategy} strategy on {args.symbol}")
    
    # Get historical data
    historical_data = get_data_for_backtest(args)
    if historical_data is None:
        return
    
    # Get strategy with parameters from command line
    strategy_params = {
        'short_window': args.short_window,
        'long_window': args.long_window,
        'threshold': args.ml_threshold,
        'retrain_period': args.retrain_period,
        # Add other parameters from CONFIG if needed
        **CONFIG.get('model', {})
    }
    
    strategy = get_strategy(args.strategy, **strategy_params)
    
    if strategy is None:
        return
    
    # Generate signals
    try:
        signals = strategy.generate_signals(historical_data)
    except Exception as e:
        logger.error(f"Error generating signals: {e}")
        if args.verbose:
            logger.error(traceback.format_exc())
        return
    
    # Run backtest
    backtester = Backtester(
        initial_capital=args.initial_capital,
        commission=args.commission,
        risk_per_trade=args.risk_per_trade,
        slippage=args.slippage
    )
    
    try:
        performance = backtester.run(signals)
    except Exception as e:
        logger.error(f"Error running backtest: {e}")
        if args.verbose:
            logger.error(traceback.format_exc())
        return
    
    if not performance:
        logger.error("Backtest failed to produce performance metrics")
        return
    
    # Print performance metrics
    logger.info("\nBacktest Performance:")
    for key, value in performance.items():
        if key in ['total_return', 'annualized_return', 'win_rate', 'max_drawdown', 'volatility']:
            logger.info(f"  {key}: {value:.2f}%")
        elif key in ['initial_equity', 'final_equity', 'max_equity', 'total_costs']:
            logger.info(f"  {key}: ${value:.2f}")
        elif key in ['sharpe_ratio', 'sortino_ratio', 'profit_factor']:
            logger.info(f"  {key}: {value:.4f}")
        elif key in ['days', 'total_trades', 'max_consecutive_wins', 'max_consecutive_losses']:
            logger.info(f"  {key}: {value}")
        # Skip detailed metrics in basic output
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run Monte Carlo analysis if requested
    if args.monte_carlo:
        try:
            logger.info("\nRunning Monte Carlo analysis...")
            mc_results = backtester.monte_carlo_analysis(iterations=args.mc_iterations)
            if mc_results:
                logger.info("\nMonte Carlo Results:")
                logger.info(f"  Median final equity: ${mc_results['median_final_equity']:.2f}")
                logger.info(f"  95% CI for final equity: ${mc_results['equity_05']:.2f} to ${mc_results['equity_95']:.2f}")
                logger.info(f"  Mean max drawdown: {mc_results['mean_max_drawdown']:.2f}%")
                logger.info(f"  Worst case drawdown: {mc_results['worst_max_drawdown']:.2f}%")
                logger.info(f"  Probability of profit: {mc_results['prob_profit']:.2f}%")
        except Exception as e:
            logger.error(f"Error in Monte Carlo analysis: {e}")
            if args.verbose:
                logger.error(traceback.format_exc())
    
    # Save results
    if args.save_results:
        # Create a directory for this specific backtest
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = os.path.join(args.output_dir, f"{args.strategy}_{args.symbol}_{timestamp}")
        os.makedirs(result_dir, exist_ok=True)
        
        # Save main results
        results_file = os.path.join(result_dir, "backtest_results.csv")
        backtester.results.to_csv(results_file)
        logger.info(f"Results saved to {results_file}")
        
        # Save trade summary
        trade_summary = backtester.get_trade_summary()
        if not trade_summary.empty:
            trades_file = os.path.join(result_dir, "trades.csv")
            trade_summary.to_csv(trades_file)
            logger.info(f"Trade summary saved to {trades_file}")
        
        # Save performance metrics
        performance_file = os.path.join(result_dir, "performance.csv")
        pd.DataFrame([performance]).to_csv(performance_file, index=False)
        logger.info(f"Performance metrics saved to {performance_file}")
        
        # Save Monte Carlo results if available
        if args.monte_carlo and 'mc_results' in locals() and mc_results:
            mc_file = os.path.join(result_dir, "monte_carlo.csv")
            pd.DataFrame([mc_results]).to_csv(mc_file, index=False)
            logger.info(f"Monte Carlo results saved to {mc_file}")
    
    # Plot results
    if not args.no_plot:
        try:
            # Plot price with signals
            signals_plot = os.path.join(args.output_dir, f"{args.strategy}_{args.symbol}_{args.timeframe}_signals.png")
            Visualizer.plot_price_with_signals(signals, signals_plot)
            
            # Plot equity and drawdown
            equity_plot = os.path.join(args.output_dir, f"{args.strategy}_{args.symbol}_{args.timeframe}_equity.png")
            backtester.plot_results(equity_plot, show_plot=False)
            
            # Plot returns distribution
            returns_plot = os.path.join(args.output_dir, f"{args.strategy}_{args.symbol}_{args.timeframe}_returns.png")
            Visualizer.plot_returns_distribution(backtester.results, returns_plot)
            
            logger.info(f"Plots saved to {args.output_dir}")
        except Exception as e:
            logger.error(f"Error creating plots: {e}")
            if args.verbose:
                logger.error(traceback.format_exc())

def run_multiple_backtests(args):
    """
    Run backtest with multiple strategies for comparison with improved error handling.
    
    Args:
        args: Command line arguments
    """
    logger.info(f"Running comparison backtest on {args.symbol}")
    
    # Get historical data
    historical_data = get_data_for_backtest(args)
    if historical_data is None:
        return
    
    # Define strategies to compare
    strategies = {
        'SMA(20,50)': get_strategy('sma', short_window=20, long_window=50),
        'SMA(10,30)': get_strategy('sma', short_window=10, long_window=30),
        'ML': get_strategy('ml', threshold=args.ml_threshold)
    }
    
    # Remove any None strategies (failed to create)
    strategies = {name: strat for name, strat in strategies.items() if strat is not None}
    
    if not strategies:
        logger.error("No valid strategies to compare")
        return
    
    # Run backtest for each strategy
    results = {}
    performances = {}
    
    for name, strategy in strategies.items():
        logger.info(f"Running backtest for {name} strategy")
        
        try:
            # Generate signals
            signals = strategy.generate_signals(historical_data)
            
            # Run backtest
            backtester = Backtester(
                initial_capital=args.initial_capital,
                commission=args.commission,
                risk_per_trade=args.risk_per_trade,
                slippage=args.slippage
            )
            performance = backtester.run(signals)
            
            if performance:
                results[name] = backtester.results
                performances[name] = performance
            else:
                logger.warning(f"Backtest for {name} failed to produce performance metrics")
        except Exception as e:
            logger.error(f"Error in backtest for {name}: {e}")
            if args.verbose:
                logger.error(traceback.format_exc())
    
    # Compare strategies
    if not results:
        logger.error("No successful backtest results to compare")
        return
        
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Plot strategy comparison
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        comparison_plot = os.path.join(args.output_dir, f"strategy_comparison_{args.symbol}_{timestamp}.png")
        Visualizer.plot_strategy_comparison(results, comparison_plot)
        logger.info(f"Comparison plot saved to {comparison_plot}")
    except Exception as e:
        logger.error(f"Error creating comparison plot: {e}")
        if args.verbose:
            logger.error(traceback.format_exc())
    
    # Print performance comparison
    logger.info("\nStrategy Performance Comparison:")
    
    # Create comparison table
    try:
        comparison = pd.DataFrame(performances).T
        
        # Select key metrics
        metrics = ['total_return', 'annualized_return', 'max_drawdown', 'sharpe_ratio', 
                   'sortino_ratio', 'total_trades', 'win_rate']
        comparison = comparison[metrics]
        
        # Format and print comparison
        comparison_str = comparison.to_string(float_format=lambda x: f"{x:.2f}")
        logger.info(f"\n{comparison_str}")
        
        # Save comparison to CSV
        if args.save_results:
            comparison_file = os.path.join(args.output_dir, f"strategy_comparison_{args.symbol}_{timestamp}.csv")
            comparison.to_csv(comparison_file)
            logger.info(f"Comparison results saved to {comparison_file}")
    except Exception as e:
        logger.error(f"Error creating comparison table: {e}")
        if args.verbose:
            logger.error(traceback.format_exc())

def run_strategy_optimization(args):
    """
    Run strategy parameter optimization.
    
    Args:
        args: Command line arguments
    """
    logger.info(f"Running strategy optimization for {args.strategy} on {args.symbol}")
    
    # Get historical data
    historical_data = get_data_for_backtest(args)
    if historical_data is None:
        return
    
    # Create strategy instance for optimization
    if args.strategy.lower() == 'sma':
        # For SMA strategy, we'll optimize the moving average windows
        strategy = SMAStrategy(short_window=args.short_window, long_window=args.long_window)
        
        # Define parameter ranges for optimization
        short_range = (5, 50, 5)  # (min, max, step)
        long_range = (20, 200, 10)
        
        try:
            # Run optimization
            logger.info("Optimizing SMA parameters...")
            optimal_params = strategy.optimize_parameters(
                historical_data, 
                short_range=short_range,
                long_range=long_range
            )
            
            if optimal_params:
                logger.info("\nOptimization Results:")
                logger.info(f"  Optimal short window: {optimal_params['short_window']}")
                logger.info(f"  Optimal long window: {optimal_params['long_window']}")
                logger.info(f"  Sharpe ratio: {optimal_params['sharpe_ratio']:.4f}")
                logger.info(f"  Total return: {optimal_params['total_return']:.2f}%")
                logger.info(f"  Max drawdown: {optimal_params['max_drawdown']:.2f}%")
                
                # Save optimization results
                if args.save_results:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    opt_file = os.path.join(args.output_dir, f"sma_optimization_{args.symbol}_{timestamp}.csv")
                    pd.DataFrame([optimal_params]).to_csv(opt_file, index=False)
                    logger.info(f"Optimization results saved to {opt_file}")
            else:
                logger.warning("Optimization did not find improved parameters")
        except Exception as e:
            logger.error(f"Error in strategy optimization: {e}")
            if args.verbose:
                logger.error(traceback.format_exc())
    else:
        logger.warning(f"Optimization not implemented for {args.strategy} strategy")

def run_paper_trading(args):
    """
    Run paper trading simulation with improved error handling.
    
    Args:
        args: Command line arguments
    """
    logger.info(f"Starting paper trading with {args.strategy} strategy on {args.symbol}")
    
    try:
        # Get strategy with parameters from command line
        strategy_params = {
            'short_window': args.short_window,
            'long_window': args.long_window,
            'threshold': args.ml_threshold,
            'retrain_period': args.retrain_period,
            # Add other parameters from CONFIG if needed
            **CONFIG.get('model', {})
        }
        
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
        logger.info("Paper trading simulation")
        logger.info("To implement full paper trading, you would:")
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
        
        # Plot recent data with signals
        if not args.no_plot:
            signals_plot = os.path.join(args.output_dir, f"paper_{args.strategy}_{args.symbol}_{args.timeframe}_signals.png")
            Visualizer.plot_price_with_signals(signals, signals_plot)
            logger.info(f"Recent signals plot saved to {signals_plot}")
    except Exception as e:
        logger.error(f"Error in paper trading: {e}")
        if args.verbose:
            logger.error(traceback.format_exc())

def confirm_live_trading():
    """Ask for confirmation before live trading."""
    print("\n⚠️  WARNING: LIVE TRADING MODE ⚠️")
    print("This will use real funds and execute actual trades.")
    print("Are you absolutely sure you want to continue?")
    
    confirm = input("Type 'yes' to confirm: ")
    return confirm.lower() == 'yes'

def run_live_trading(args):
    """
    Run live trading with specified parameters and improved safety checks.
    
    Args:
        args: Command line arguments
    """
    # Confirm live trading
    if not confirm_live_trading():
        logger.info("Live trading cancelled")
        return
    
    logger.info(f"Starting live trading with {args.strategy} strategy on {args.symbol}")
    logger.info("Live trading not implemented in this version")
    logger.info("To implement live trading, you would:")
    logger.info("1. Connect to exchange API with authentication")
    logger.info("2. Implement order placement logic")
    logger.info("3. Implement risk management")
    logger.info("4. Implement position tracking")
    logger.info("5. Implement error handling and failsafes")

def main():
    """Main entry point for the trading bot with improved error handling."""
    # Parse command-line arguments
    args = parse_arguments()
    
    # Set up signal handler for graceful shutdown
    setup_signal_handler()
    
    try:
        # Route to appropriate function based on mode
        if args.mode == 'backtest':
            run_backtest(args)
        elif args.mode == 'compare':
            run_multiple_backtests(args)
        elif args.mode == 'optimize':
            run_strategy_optimization(args)
        elif args.mode == 'paper':
            run_paper_trading(args)
        elif args.mode == 'live':
            run_live_trading(args)
        elif args.mode == 'simulated':
            logger.info("Simulated mode: Using simulated data")
            args.data_source = 'simulated'
            run_backtest(args)
        else:
            logger.error(f"Unknown mode: {args.mode}")
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Error running bot: {e}")
        if args.verbose:
            logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()