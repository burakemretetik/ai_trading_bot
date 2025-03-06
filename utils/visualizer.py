# utils/visualizer.py - Simplified visualization tools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import logging
import os

logger = logging.getLogger(__name__)

class Visualizer:
    """Class for creating trading-related visualizations."""
    
    @staticmethod
    def plot_price_with_signals(df, save_path=None):
        """
        Plot price chart with buy/sell signals.
        
        Args:
            df: DataFrame with price data and signals
            save_path: Path to save the plot
        """
        # Check required columns
        if 'close' not in df.columns or 'position' not in df.columns:
            logger.error("DataFrame missing required columns for signal plotting")
            return
        
        plt.figure(figsize=(12, 6))
        
        # Plot price
        plt.plot(df.index, df['close'], label='Close Price', color='blue')
        
        # Plot buy signals
        buy_signals = df[df['position'] == 2]  # Position changed from 0 to 1 or -1 to 1
        if not buy_signals.empty:
            plt.scatter(buy_signals.index, buy_signals['close'], marker='^', color='green', s=100, label='Buy')
        
        # Plot sell signals
        sell_signals = df[df['position'] == -2]  # Position changed from 0 to -1 or 1 to -1
        if not sell_signals.empty:
            plt.scatter(sell_signals.index, sell_signals['close'], marker='v', color='red', s=100, label='Sell')
        
        # Add moving averages if available
        if 'short_ma' in df.columns and 'long_ma' in df.columns:
            plt.plot(df.index, df['short_ma'], color='orange', label='Short MA')
            plt.plot(df.index, df['long_ma'], color='purple', label='Long MA')
        
        # Add labels and legend
        plt.title('Price Chart with Trading Signals')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        # Save or show the plot
        if save_path:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            logger.info(f"Price chart saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    @staticmethod
    def plot_returns_distribution(df, save_path=None):
        """
        Plot distribution of returns.
        
        Args:
            df: DataFrame with returns
            save_path: Path to save the plot
        """
        # Check required columns
        if 'returns' not in df.columns:
            logger.error("DataFrame missing 'returns' column")
            return
        
        plt.figure(figsize=(10, 6))
        
        # Plot histogram
        plt.hist(df['returns'].dropna() * 100, bins=50, alpha=0.7, color='blue')
        
        # Add normal distribution curve
        from scipy import stats
        import numpy as np
        returns = df['returns'].dropna() * 100
        mu, std = returns.mean(), returns.std()
        x = np.linspace(mu - 3*std, mu + 3*std, 100)
        p = stats.norm.pdf(x, mu, std)
        plt.plot(x, p * len(returns) * (returns.max() - returns.min()) / 50, 'r-', linewidth=2)
        
        # Add vertical line at zero
        plt.axvline(x=0, color='black', linestyle='--')
        
        # Add labels
        plt.title('Distribution of Daily Returns')
        plt.xlabel('Return (%)')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.tight_layout()
        
        # Save or show the plot
        if save_path:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            logger.info(f"Returns distribution chart saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    @staticmethod
    def plot_strategy_comparison(strategies_results, save_path=None):
        """
        Plot comparison of multiple strategies.
        
        Args:
            strategies_results: Dictionary with strategy names as keys and DataFrames as values
            save_path: Path to save the plot
        """
        plt.figure(figsize=(12, 6))
        
        # Plot equity curve for each strategy
        for strategy_name, results in strategies_results.items():
            if 'total_equity' in results.columns:
                # Normalize to starting value of 100 for fair comparison
                normalized_equity = results['total_equity'] / results['total_equity'].iloc[0] * 100
                plt.plot(results.index, normalized_equity, label=strategy_name)
            else:
                logger.warning(f"Strategy {strategy_name} missing 'total_equity' column")
        
        # Add labels and legend
        plt.title('Strategy Performance Comparison')
        plt.xlabel('Date')
        plt.ylabel('Normalized Value (starting at 100)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        # Save or show the plot
        if save_path:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            logger.info(f"Strategy comparison chart saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    @staticmethod
    def plot_equity_and_drawdown(df, save_path=None):
        """
        Plot equity curve and drawdown.
        
        Args:
            df: DataFrame with equity and drawdown
            save_path: Path to save the plot
        """
        # Check required columns
        if 'total_equity' not in df.columns or 'drawdown' not in df.columns:
            logger.error("DataFrame missing required columns")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # Plot equity curve
        ax1.plot(df.index, df['total_equity'], label='Portfolio Value')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.set_title('Equity Curve')
        ax1.legend()
        ax1.grid(True)
        
        # Plot drawdown
        ax2.fill_between(df.index, df['drawdown'], 0, color='red', alpha=0.3, label='Drawdown (%)')
        ax2.set_ylabel('Drawdown (%)')
        ax2.set_xlabel('Date')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        # Save or show the plot
        if save_path:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            logger.info(f"Equity and drawdown chart saved to {save_path}")
        else:
            plt.show()
        
        plt.close()