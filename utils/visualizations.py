# utils/visualizations.py - Visualization tools for trading analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
import logging

logger = logging.getLogger(__name__)

class TradingVisualizer:
    """Class for creating trading-related visualizations."""
    
    @staticmethod
    def format_currency(x, pos):
        """Format y-axis labels as currency."""
        return f"${x:,.2f}"
    
    @staticmethod
    def plot_price_with_signals(df, title="Price Chart with Trading Signals", save_path=None):
        """
        Plot price chart with buy/sell signals.
        
        Args:
            df (pandas.DataFrame): DataFrame with price data and signals
            title (str): Plot title
            save_path (str, optional): Path to save the plot
        """
        if 'close' not in df.columns or 'position' not in df.columns:
            logger.error("DataFrame missing required columns for signal plotting")
            return
        
        plt.figure(figsize=(15, 8))
        
        # Format the plot
        plt.style.use('seaborn-v0_8-darkgrid')
        ax = plt.gca()
        ax.yaxis.set_major_formatter(FuncFormatter(TradingVisualizer.format_currency))
        
        # Plot price
        plt.plot(df.index, df['close'], label='Close Price', color='blue', alpha=0.7)
        
        # Plot buy signals
        buy_signals = df[df['position'] == 2]
        plt.scatter(buy_signals.index, buy_signals['close'], marker='^', color='green', s=100, label='Buy Signal')
        
        # Plot sell signals
        sell_signals = df[df['position'] == -2]
        plt.scatter(sell_signals.index, sell_signals['close'], marker='v', color='red', s=100, label='Sell Signal')
        
        # Add moving averages if available
        if 'short_ma' in df.columns and 'long_ma' in df.columns:
            plt.plot(df.index, df['short_ma'], color='orange', label=f"Short MA ({df['short_ma'].name.split('_')[1]})")
            plt.plot(df.index, df['long_ma'], color='purple', label=f"Long MA ({df['long_ma'].name.split('_')[1]})")
        
        # Add predicted price if available
        if 'predicted_price' in df.columns:
            plt.plot(df.index, df['predicted_price'], color='red', linestyle='--', label='Predicted Price')
            
        # Format date axis
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
        
        # Add labels and legend
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        # Save or show the plot
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Price chart with signals saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    @staticmethod
    def plot_portfolio_performance(df, initial_capital, title="Portfolio Performance", save_path=None):
        """
        Plot portfolio performance metrics.
        
        Args:
            df (pandas.DataFrame): DataFrame with trading results
            initial_capital (float): Initial capital
            title (str): Plot title
            save_path (str, optional): Path to save the plot
        """
        if 'total_equity' not in df.columns:
            logger.error("DataFrame missing required columns for performance plotting")
            return
        
        # Create subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12), sharex=True, gridspec_kw={'height_ratios': [3, 1, 1]})
        
        # Format the plot
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # Plot 1: Equity curve
        ax1.plot(df.index, df['total_equity'], label='Portfolio Value', color='blue')
        ax1.axhline(y=initial_capital, color='gray', linestyle='--', label='Initial Capital')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.set_title(title)
        ax1.yaxis.set_major_formatter(FuncFormatter(TradingVisualizer.format_currency))
        ax1.legend()
        ax1.grid(True)
        
        # Plot 2: Returns
        ax2.plot(df.index, df['returns'].rolling(window=20).mean() * 100, label='20-Day Rolling Return (%)', color='green')
        ax2.axhline(y=0, color='gray', linestyle='--')
        ax2.set_ylabel('Return (%)')
        ax2.legend()
        ax2.grid(True)
        
        # Plot 3: Drawdown
        if 'drawdown' in df.columns:
            ax3.fill_between(df.index, df['drawdown'], 0, color='red', alpha=0.3, label='Drawdown (%)')
            ax3.set_ylabel('Drawdown (%)')
            ax3.set_xlabel('Date')
            ax3.legend()
            ax3.grid(True)
        
        # Format date axis
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax3.xaxis.set_major_locator(mdates.AutoDateLocator())
        
        plt.tight_layout()
        
        # Save or show the plot
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Portfolio performance chart saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    @staticmethod
    def plot_feature_importance(feature_importance, title="Feature Importance", save_path=None):
        """
        Plot feature importance from ML model.
        
        Args:
            feature_importance (pandas.DataFrame): DataFrame with feature names and importance scores
            title (str): Plot title
            save_path (str, optional): Path to save the plot
        """
        if 'feature' not in feature_importance.columns or 'importance' not in feature_importance.columns:
            logger.error("DataFrame missing required columns for feature importance plotting")
            return
        
        # Sort by importance
        df = feature_importance.sort_values('importance', ascending=True)
        
        # Only show top N features if there are many
        if len(df) > 15:
            df = df.tail(15)
        
        plt.figure(figsize=(10, 8))
        
        # Format the plot
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # Create horizontal bar chart
        plt.barh(df['feature'], df['importance'], color='skyblue')
        
        # Add feature importance values
        for i, v in enumerate(df['importance']):
            plt.text(v + 0.001, i, f"{v:.4f}", va='center')
        
        # Add labels and title
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title(title)
        plt.tight_layout()
        
        # Save or show the plot
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Feature importance chart saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    @staticmethod
    def plot_correlation_matrix(df, title="Feature Correlation Matrix", save_path=None):
        """
        Plot correlation matrix of features.
        
        Args:
            df (pandas.DataFrame): DataFrame with features
            title (str): Plot title
            save_path (str, optional): Path to save the plot
        """
        # Calculate correlation matrix
        corr = df.corr()
        
        plt.figure(figsize=(12, 10))
        
        # Create heatmap
        mask = np.triu(np.ones_like(corr, dtype=bool))
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        
        sns.heatmap(
            corr, 
            mask=mask, 
            cmap=cmap, 
            vmax=1, 
            vmin=-1, 
            center=0,
            annot=False, 
            square=True, 
            linewidths=.5, 
            cbar_kws={"shrink": .5}
        )
        
        plt.title(title)
        plt.tight_layout()
        
        # Save or show the plot
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Correlation matrix saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    @staticmethod
    def plot_equity_curve_comparison(results_dict, title="Strategy Comparison", save_path=None):
        """
        Plot equity curves for multiple strategies on the same chart.
        
        Args:
            results_dict (dict): Dictionary of DataFrames with equity curves for each strategy
            title (str): Plot title
            save_path (str, optional): Path to save the plot
        """
        plt.figure(figsize=(15, 8))
        
        # Format the plot
        plt.style.use('seaborn-v0_8-darkgrid')
        ax = plt.gca()
        ax.yaxis.set_major_formatter(FuncFormatter(TradingVisualizer.format_currency))
        
        # Plot equity curve for each strategy
        for strategy_name, df in results_dict.items():
            if 'total_equity' in df.columns:
                plt.plot(df.index, df['total_equity'], label=f"{strategy_name}")
            else:
                logger.warning(f"DataFrame for {strategy_name} missing 'total_equity' column")
        
        # Add labels and legend
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value ($)')
        plt.legend()
        plt.grid(True)
        
        # Format date axis
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
        
        plt.tight_layout()
        
        # Save or show the plot
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Strategy comparison chart saved to {save_path}")
        else:
            plt.show()
        
        plt.close()