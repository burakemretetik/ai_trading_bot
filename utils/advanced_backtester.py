# utils/advanced_backtester.py - Advanced backtesting with walk-forward testing and Monte Carlo simulation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Union, Any, Callable, Type
import logging
import os
from datetime import datetime, timedelta
import time
from tqdm import tqdm

from utils.constants import PerformanceMetrics
from utils.backtester import Backtester

logger = logging.getLogger(__name__)

class WalkForwardTest:
    """Class for performing walk-forward backtesting."""
    
    def __init__(self, 
               strategy_class: Any,
               train_size: int = 252,  # Default: 1 year of trading days
               test_size: int = 63,    # Default: 3 months of trading days
               step_size: int = 63,    # Default: 3 months of trading days
               train_test_gap: int = 0,
               initial_capital: float = 10000.0,
               commission: float = 0.001):
        """
        Initialize walk-forward test.
        
        Args:
            strategy_class: Strategy class to test
            train_size: Number of bars in each training window
            test_size: Number of bars in each testing window
            step_size: Number of bars to shift window forward
            train_test_gap: Number of bars to skip between train and test
            initial_capital: Initial capital for backtesting
            commission: Commission rate for trades
        """
        self.strategy_class = strategy_class
        self.train_size = train_size
        self.test_size = test_size
        self.step_size = step_size
        self.train_test_gap = train_test_gap
        self.initial_capital = initial_capital
        self.commission = commission
        
        self.results = None
        self.windows = None
        self.performance_metrics = None
    
    def run(self, 
          data: pd.DataFrame, 
          strategy_params: Optional[Dict[str, Any]] = None,
          optimize_func: Optional[Callable] = None,
          verbose: bool = True) -> pd.DataFrame:
        """
        Run walk-forward test.
        
        Args:
            data: OHLCV DataFrame
            strategy_params: Fixed strategy parameters (if not optimizing)
            optimize_func: Function to optimize strategy parameters for each window
            verbose: Whether to print progress
            
        Returns:
            DataFrame with concatenated test results
        """
        if len(data) < self.train_size + self.train_test_gap + self.test_size:
            logger.error(f"Not enough data for walk-forward test. Need at least {self.train_size + self.train_test_gap + self.test_size} bars.")
            return pd.DataFrame()
        
        logger.info("Starting walk-forward test")
        start_time = time.time()
        
        # Create windows
        self.windows = self._create_windows(data)
        
        if not self.windows:
            logger.error("Failed to create windows for walk-forward test")
            return pd.DataFrame()
        
        logger.info(f"Created {len(self.windows)} train-test windows")
        
        # Initialize results containers
        all_test_results = []
        self.performance_metrics = []
        
        # Process each window
        for i, window in enumerate(tqdm(self.windows, desc="Processing windows", disable=not verbose)):
            train_start, train_end, test_start, test_end = window
            
            if verbose:
                logger.info(f"Window {i+1}/{len(self.windows)}: "
                         f"Train {train_start.strftime('%Y-%m-%d')} to {train_end.strftime('%Y-%m-%d')}, "
                         f"Test {test_start.strftime('%Y-%m-%d')} to {test_end.strftime('%Y-%m-%d')}")
            
            # Get train and test data
            train_data = data.loc[train_start:train_end].copy()
            test_data = data.loc[test_start:test_end].copy()
            
            # Skip window if not enough data
            if len(train_data) < self.train_size * 0.8 or len(test_data) < self.test_size * 0.8:
                logger.warning(f"Skipping window {i+1} due to insufficient data: "
                            f"Train size: {len(train_data)}, Test size: {len(test_data)}")
                continue
            
            # Get strategy parameters
            if optimize_func is not None:
                # Optimize strategy for this window
                window_params = optimize_func(train_data)
                
                if verbose:
                    logger.info(f"Optimized parameters for window {i+1}: {window_params}")
            else:
                # Use fixed parameters
                window_params = strategy_params or {}
            
            # Initialize strategy
            strategy = self.strategy_class(**window_params)
            
            # Train strategy on training data
            if hasattr(strategy, 'train_model'):
                strategy.train_model(train_data)
            
            # Generate signals on test data
            signals = strategy.generate_signals(test_data)
            
            # Run backtest
            backtester = Backtester(
                initial_capital=self.initial_capital,
                commission=self.commission
            )
            
            performance = backtester.run(signals)
            
            # Store results with window information
            signals['window'] = i + 1
            signals['window_type'] = 'test'
            all_test_results.append(signals)
            
            # Store performance metrics
            performance['window'] = i + 1
            performance['train_start'] = train_start
            performance['train_end'] = train_end
            performance['test_start'] = test_start
            performance['test_end'] = test_end
            self.performance_metrics.append(performance)
        
        # Combine all test results
        if all_test_results:
            self.results = pd.concat(all_test_results)
            
            if verbose:
                logger.info(f"Walk-forward test completed in {time.time() - start_time:.2f} seconds")
                logger.info(f"Final equity: ${self.results['total_equity'].iloc[-1]:.2f}")
                
                # Calculate overall performance metrics
                initial_equity = self.initial_capital
                final_equity = self.results['total_equity'].iloc[-1]
                total_return = (final_equity - initial_equity) / initial_equity * 100
                
                logger.info(f"Overall return: {total_return:.2f}%")
                
                # Calculate window-specific statistics
                returns_by_window = []
                for i in range(1, len(self.windows) + 1):
                    window_results = self.results[self.results['window'] == i]
                    if not window_results.empty:
                        window_return = (window_results['total_equity'].iloc[-1] - window_results['total_equity'].iloc[0]) / window_results['total_equity'].iloc[0] * 100
                        returns_by_window.append(window_return)
                
                if returns_by_window:
                    logger.info(f"Average window return: {np.mean(returns_by_window):.2f}%")
                    logger.info(f"Win rate (windows): {np.mean([r > 0 for r in returns_by_window]):.2f}")
            
            return self.results
        else:
            logger.error("No valid windows processed in walk-forward test")
            return pd.DataFrame()
    
    def _create_windows(self, data: pd.DataFrame) -> List[Tuple[datetime, datetime, datetime, datetime]]:
        """
        Create train-test windows for walk-forward testing.
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            List of (train_start, train_end, test_start, test_end) tuples
        """
        windows = []
        
        # Get date range
        dates = data.index
        
        # Initialize first window
        train_start_idx = 0
        
        while True:
            # Calculate indices
            train_end_idx = train_start_idx + self.train_size - 1
            test_start_idx = train_end_idx + 1 + self.train_test_gap
            test_end_idx = test_start_idx + self.test_size - 1
            
            # Check if we have enough data for this window
            if test_end_idx >= len(dates):
                break
            
            # Get actual dates
            train_start = dates[train_start_idx]
            train_end = dates[train_end_idx]
            test_start = dates[test_start_idx]
            test_end = dates[test_end_idx]
            
            # Add window
            windows.append((train_start, train_end, test_start, test_end))
            
            # Move to next window
            train_start_idx += self.step_size
        
        return windows
    
    def plot_results(self, 
                   figsize: Tuple[int, int] = (12, 8),
                   include_individual_windows: bool = True,
                   save_path: Optional[str] = None) -> None:
        """
        Plot walk-forward test results.
        
        Args:
            figsize: Figure size
            include_individual_windows: Whether to plot individual window results
            save_path: Path to save the plot
        """
        if self.results is None:
            logger.error("No results available. Run walk-forward test first.")
            return
        
        plt.figure(figsize=figsize)
        
        # Plot combined equity curve
        plt.plot(self.results.index, self.results['total_equity'], 'b-', label='Equity Curve')
        
        # Plot window boundaries
        if include_individual_windows and self.windows:
            for i, window in enumerate(self.windows):
                _, _, test_start, test_end = window
                
                # Plot test window boundaries
                plt.axvline(x=test_start, color='g', linestyle='--', alpha=0.3)
                plt.axvline(x=test_end, color='r', linestyle='--', alpha=0.3)
                
                # Add window number
                plt.text(test_start, self.results['total_equity'].max() * 0.95, 
                      f"W{i+1}", fontsize=8)
        
        # Plot initial capital
        plt.axhline(y=self.initial_capital, color='k', linestyle=':', label='Initial Capital')
        
        # Add title and labels
        plt.title('Walk-Forward Test Results', fontsize=16)
        plt.xlabel('Date', fontsize=14)
        plt.ylabel('Equity ($)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Format currency on y-axis
        from matplotlib.ticker import FuncFormatter
        ax = plt.gca()
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:,.0f}'))
        
        plt.tight_layout()
        
        # Save or show the plot
        if save_path:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            logger.info(f"Walk-forward test results plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_window_returns(self, 
                        figsize: Tuple[int, int] = (12, 6),
                        save_path: Optional[str] = None) -> None:
        """
        Plot returns by window.
        
        Args:
            figsize: Figure size
            save_path: Path to save the plot
        """
        if not self.performance_metrics:
            logger.error("No performance metrics available. Run walk-forward test first.")
            return
        
        # Calculate returns by window
        window_returns = []
        
        for window in range(1, len(self.windows) + 1):
            window_results = self.results[self.results['window'] == window]
            
            if not window_results.empty:
                initial_equity = window_results['total_equity'].iloc[0]
                final_equity = window_results['total_equity'].iloc[-1]
                window_return = (final_equity - initial_equity) / initial_equity * 100
                
                window_returns.append({
                    'window': window,
                    'return': window_return,
                    'test_start': self.windows[window-1][2],
                    'test_end': self.windows[window-1][3]
                })
        
        if not window_returns:
            logger.error("No window returns available for plotting.")
            return
        
        # Create DataFrame
        df = pd.DataFrame(window_returns)
        
        plt.figure(figsize=figsize)
        
        # Create bar chart
        bars = plt.bar(df['window'], df['return'], color=df['return'].apply(lambda x: 'green' if x > 0 else 'red'))
        
        # Add values to bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., 
                  height + (1 if height > 0 else -3),
                  f'{height:.1f}%',
                  ha='center', va='bottom' if height > 0 else 'top',
                  fontsize=8)
        
        # Add title and labels
        plt.title('Returns by Window', fontsize=16)
        plt.xlabel('Window', fontsize=14)
        plt.ylabel('Return (%)', fontsize=14)
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add zero line
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        
        # Add average line
        avg_return = df['return'].mean()
        plt.axhline(y=avg_return, color='b', linestyle='--', label=f'Average: {avg_return:.1f}%')
        
        plt.legend()
        plt.tight_layout()
        
        # Save or show the plot
        if save_path:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            logger.info(f"Window returns plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()


class MonteCarloBacktest:
    """Class for Monte Carlo backtesting simulation."""
    
    def __init__(self, 
               initial_capital: float = 10000.0, 
               commission: float = 0.001,
               n_simulations: int = 1000,
               confidence_level: float = 0.95):
        """
        Initialize Monte Carlo backtest.
        
        Args:
            initial_capital: Initial capital for backtesting
            commission: Commission rate for trades
            n_simulations: Number of Monte Carlo simulations
            confidence_level: Confidence level for risk metrics
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.n_simulations = n_simulations
        self.confidence_level = confidence_level
        
        self.original_results = None
        self.simulation_results = None
        self.risk_metrics = None
    
    def run(self, 
          signals: pd.DataFrame, 
          method: str = 'bootstrap',
          block_size: int = 10,
          verbose: bool = True) -> Dict[str, Any]:
        """
        Run Monte Carlo simulations.
        
        Args:
            signals: DataFrame with strategy signals
            method: Simulation method ('bootstrap', 'block_bootstrap', or 'random_walk')
            block_size: Block size for block bootstrap
            verbose: Whether to print progress
            
        Returns:
            Dictionary with risk metrics
        """
        logger.info(f"Running {self.n_simulations} Monte Carlo simulations using {method} method")
        start_time = time.time()
        
        # Store original signals
        self.original_results = signals.copy()
        
        # Extract trade returns
        if 'returns' not in self.original_results.columns:
            logger.error("No returns column found in signals DataFrame")
            return {}
        
        returns = self.original_results['returns'].dropna()
        
        if len(returns) == 0:
            logger.error("No valid returns found in signals DataFrame")
            return {}
        
        # Run simulations
        if method == 'bootstrap':
            simulations = self._bootstrap_simulation(returns)
        elif method == 'block_bootstrap':
            simulations = self._block_bootstrap_simulation(returns, block_size)
        elif method == 'random_walk':
            simulations = self._random_walk_simulation(returns)
        else:
            logger.error(f"Unknown simulation method: {method}")
            return {}
        
        # Calculate equity curves
        self.simulation_results = self._calculate_equity_curves(simulations)
        
        # Calculate risk metrics
        self.risk_metrics = self._calculate_risk_metrics()
        
        if verbose:
            logger.info(f"Monte Carlo simulation completed in {time.time() - start_time:.2f} seconds")
            logger.info(f"Risk metrics at {self.confidence_level*100}% confidence level:")
            for key, value in self.risk_metrics.items():
                logger.info(f"  {key}: {value}")
        
        return self.risk_metrics
    
    def _bootstrap_simulation(self, returns: pd.Series) -> np.ndarray:
        """
        Run bootstrap simulation.
        
        Args:
            returns: Series of returns
            
        Returns:
            Array of simulated returns
        """
        # Convert returns to numpy array
        returns_array = returns.values
        
        # Generate bootstrap samples
        n_returns = len(returns_array)
        
        # Generate random indices for each simulation
        indices = np.random.randint(0, n_returns, size=(self.n_simulations, n_returns))
        
        # Generate simulated returns
        simulations = returns_array[indices]
        
        return simulations
    
    def _block_bootstrap_simulation(self, 
                               returns: pd.Series, 
                               block_size: int) -> np.ndarray:
        """
        Run block bootstrap simulation.
        
        Args:
            returns: Series of returns
            block_size: Block size
            
        Returns:
            Array of simulated returns
        """
        # Convert returns to numpy array
        returns_array = returns.values
        
        # Generate block bootstrap samples
        n_returns = len(returns_array)
        n_blocks = int(np.ceil(n_returns / block_size))
        
        # Create simulated returns array
        simulations = np.zeros((self.n_simulations, n_returns))
        
        # Generate simulations
        for i in range(self.n_simulations):
            # Generate random starting indices for blocks
            start_indices = np.random.randint(0, n_returns - block_size + 1, size=n_blocks)
            
            # Fill in simulated returns with blocks
            sim_returns = []
            
            for start_idx in start_indices:
                sim_returns.extend(returns_array[start_idx:start_idx + block_size])
            
            # Trim to correct length
            sim_returns = sim_returns[:n_returns]
            
            simulations[i] = sim_returns
        
        return simulations
    
    def _random_walk_simulation(self, returns: pd.Series) -> np.ndarray:
        """
        Run random walk simulation.
        
        Args:
            returns: Series of returns
            
        Returns:
            Array of simulated returns
        """
        # Calculate mean and standard deviation of returns
        mean_return = returns.mean()
        std_return = returns.std()
        
        # Generate random walk samples
        n_returns = len(returns)
        
        # Generate simulated returns
        simulations = np.random.normal(mean_return, std_return, size=(self.n_simulations, n_returns))
        
        return simulations
    
    def _calculate_equity_curves(self, simulations: np.ndarray) -> np.ndarray:
        """
        Calculate equity curves from simulated returns.
        
        Args:
            simulations: Array of simulated returns
            
        Returns:
            Array of equity curves
        """
        # Calculate cumulative returns
        cum_returns = np.cumprod(1 + simulations, axis=1)
        
        # Calculate equity curves
        equity_curves = self.initial_capital * cum_returns
        
        return equity_curves
    
    def _calculate_risk_metrics(self) -> Dict[str, float]:
        """
        Calculate risk metrics.
        
        Returns:
            Dictionary with risk metrics
        """
        # Extract final equity values
        final_equity = self.simulation_results[:, -1]
        
        # Calculate returns
        total_returns = (final_equity - self.initial_capital) / self.initial_capital
        
        # Calculate confidence intervals
        alpha = 1 - self.confidence_level
        lower_idx = int(alpha / 2 * self.n_simulations)
        upper_idx = int((1 - alpha / 2) * self.n_simulations)
        
        # Sort returns for percentiles
        sorted_returns = np.sort(total_returns)
        
        # Calculate metrics
        metrics = {
            'expected_return': np.mean(total_returns) * 100,
            'return_std': np.std(total_returns) * 100,
            'var_95': -np.percentile(total_returns, 5) * 100,  # Value at Risk at 95% confidence
            'return_lower_bound': sorted_returns[lower_idx] * 100,
            'return_upper_bound': sorted_returns[upper_idx] * 100,
            'probability_of_profit': np.mean(total_returns > 0) * 100,
            'probability_of_loss': np.mean(total_returns < 0) * 100,
            'max_return': np.max(total_returns) * 100,
            'min_return': np.min(total_returns) * 100,
            'median_return': np.median(total_returns) * 100
        }
        
        # Calculate drawdowns for each simulation
        max_drawdowns = []
        
        for i in range(self.n_simulations):
            equity_curve = self.simulation_results[i]
            running_max = np.maximum.accumulate(equity_curve)
            drawdown = (equity_curve - running_max) / running_max * 100
            max_drawdowns.append(np.min(drawdown))
        
        # Add drawdown metrics
        metrics['expected_max_drawdown'] = np.mean(max_drawdowns)
        metrics['max_drawdown_worst_case'] = np.min(max_drawdowns)
        metrics['max_drawdown_95'] = np.percentile(max_drawdowns, 5)
        
        return metrics
    
    def plot_simulations(self, 
                      figsize: Tuple[int, int] = (12, 8),
                      max_sims_to_plot: int = 100,
                      highlight_original: bool = True,
                      show_confidence: bool = True,
                      save_path: Optional[str] = None) -> None:
        """
        Plot Monte Carlo simulations.
        
        Args:
            figsize: Figure size
            max_sims_to_plot: Maximum number of simulations to plot
            highlight_original: Whether to highlight original results
            show_confidence: Whether to show confidence interval
            save_path: Path to save the plot
        """
        if self.simulation_results is None:
            logger.error("No simulation results available. Run Monte Carlo simulation first.")
            return
        
        plt.figure(figsize=figsize)
        
        # Plot simulations (limit to max_sims_to_plot)
        n_to_plot = min(max_sims_to_plot, self.n_simulations)
        indices = np.random.choice(self.n_simulations, n_to_plot, replace=False)
        
        for i in indices:
            plt.plot(self.simulation_results[i], color='blue', alpha=0.1)
        
        # Plot confidence interval
        if show_confidence:
            lower = np.percentile(self.simulation_results, (1 - self.confidence_level) / 2 * 100, axis=0)
            upper = np.percentile(self.simulation_results, (1 + self.confidence_level) / 2 * 100, axis=0)
            median = np.percentile(self.simulation_results, 50, axis=0)
            
            plt.plot(median, color='blue', linewidth=2, label='Median')
            plt.fill_between(range(len(lower)), lower, upper, color='blue', alpha=0.2, 
                          label=f'{self.confidence_level*100:.0f}% Confidence Interval')
        
        # Plot original results
        if highlight_original and 'total_equity' in self.original_results.columns:
            plt.plot(self.original_results['total_equity'], color='red', linewidth=2, label='Original')
        
        # Add title and labels
        plt.title('Monte Carlo Simulation Results', fontsize=16)
        plt.xlabel('Trading Period', fontsize=14)
        plt.ylabel('Equity ($)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Format currency on y-axis
        from matplotlib.ticker import FuncFormatter
        ax = plt.gca()
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:,.0f}'))
        
        plt.tight_layout()
        
        # Save or show the plot
        if save_path:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            logger.info(f"Monte Carlo simulation plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_return_distribution(self, 
                               figsize: Tuple[int, int] = (12, 6),
                               save_path: Optional[str] = None) -> None:
        """
        Plot return distribution.
        
        Args:
            figsize: Figure size
            save_path: Path to save the plot
        """
        if self.simulation_results is None:
            logger.error("No simulation results available. Run Monte Carlo simulation first.")
            return
        
        # Calculate returns
        final_equity = self.simulation_results[:, -1]
        total_returns = (final_equity - self.initial_capital) / self.initial_capital * 100
        
        plt.figure(figsize=figsize)
        
        # Plot histogram
        plt.hist(total_returns, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        
        # Add vertical lines for key metrics
        plt.axvline(x=np.mean(total_returns), color='red', linestyle='--', 
                 label=f'Mean: {np.mean(total_returns):.2f}%')
        
        plt.axvline(x=np.median(total_returns), color='green', linestyle='--', 
                 label=f'Median: {np.median(total_returns):.2f}%')
        
        # Add confidence interval
        lower = np.percentile(total_returns, (1 - self.confidence_level) / 2 * 100)
        upper = np.percentile(total_returns, (1 + self.confidence_level) / 2 * 100)
        
        plt.axvline(x=lower, color='blue', linestyle=':', 
                 label=f'{(1 - self.confidence_level) / 2 * 100:.1f}th percentile: {lower:.2f}%')
        
        plt.axvline(x=upper, color='blue', linestyle=':', 
                 label=f'{(1 + self.confidence_level) / 2 * 100:.1f}th percentile: {upper:.2f}%')
        
        # Add title and labels
        plt.title('Monte Carlo Simulation: Return Distribution', fontsize=16)
        plt.xlabel('Total Return (%)', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        
        # Save or show the plot
        if save_path:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            logger.info(f"Return distribution plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_drawdown_distribution(self, 
                                figsize: Tuple[int, int] = (12, 6),
                                save_path: Optional[str] = None) -> None:
        """
        Plot drawdown distribution.
        
        Args:
            figsize: Figure size
            save_path: Path to save the plot
        """
        if self.simulation_results is None:
            logger.error("No simulation results available. Run Monte Carlo simulation first.")
            return
        
        # Calculate drawdowns
        max_drawdowns = []
        
        for i in range(self.n_simulations):
            equity_curve = self.simulation_results[i]
            running_max = np.maximum.accumulate(equity_curve)
            drawdown = (equity_curve - running_max) / running_max * 100
            max_drawdowns.append(np.min(drawdown))
        
        plt.figure(figsize=figsize)
        
        # Plot histogram
        plt.hist(max_drawdowns, bins=50, alpha=0.7, color='salmon', edgecolor='black')
        
        # Add vertical lines for key metrics
        plt.axvline(x=np.mean(max_drawdowns), color='red', linestyle='--', 
                 label=f'Mean: {np.mean(max_drawdowns):.2f}%')
        
        plt.axvline(x=np.percentile(max_drawdowns, 5), color='blue', linestyle=':', 
                 label=f'95th percentile: {np.percentile(max_drawdowns, 5):.2f}%')
        
        # Add title and labels
        plt.title('Monte Carlo Simulation: Maximum Drawdown Distribution', fontsize=16)
        plt.xlabel('Maximum Drawdown (%)', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        
        # Save or show the plot
        if save_path:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            logger.info(f"Drawdown distribution plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()