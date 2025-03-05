# utils/backtester.py - Backtesting framework
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from config import INITIAL_CAPITAL, COMMISSION
import logging

logger = logging.getLogger(__name__)

class Backtester:
    """Class for backtesting trading strategies."""
    
    def __init__(self, initial_capital=INITIAL_CAPITAL, commission=COMMISSION):
        """Initialize backtester with starting capital and commission rate."""
        self.initial_capital = initial_capital
        self.commission = commission
        self.results = None
    
    def run(self, signals):
        """Run backtest on the signals dataframe."""
        logger.info(f"Running backtest with initial capital: ${self.initial_capital}")
        
        # Make a copy of the signals dataframe
        df = signals.copy()
        
        # Initialize portfolio metrics
        df['position_value'] = 0.0
        df['holdings'] = 0.0
        df['cash'] = 0.0
        df['total_equity'] = 0.0
        df['returns'] = 0.0
        
        # Calculate positions and holdings
        df['position_size'] = df['position'].apply(lambda x: 1 if x == 2 else -1 if x == -2 else 0)
        df['position_value'] = df['position_size'] * df['close']
        
        # Calculate cash and holdings
        df['holdings'] = df['position_value'].cumsum()
        
        # Calculate transaction costs
        df['transaction_cost'] = abs(df['position_value']) * self.commission
        df['transaction_cost'] = df['transaction_cost'].fillna(0)
        
        # Calculate cash balance
        df['cash'] = self.initial_capital - df['holdings'] - df['transaction_cost'].cumsum()
        
        # Calculate total equity and returns
        df['total_equity'] = df['cash'] + df['holdings']
        df['returns'] = df['total_equity'].pct_change()
        
        # Store results
        self.results = df
        
        # Calculate performance metrics
        performance = self._calculate_performance()
        
        logger.info(f"Backtest completed. Final equity: ${performance['final_equity']:.2f}")
        
        return performance
    
    def _calculate_performance(self):
        """Calculate performance metrics."""
        if self.results is None:
            return {}
        
        df = self.results
        
        # Calculate basic metrics
        initial_equity = self.initial_capital
        final_equity = df['total_equity'].iloc[-1]
        total_return = (final_equity - initial_equity) / initial_equity * 100
        
        # Calculate sharpe ratio (annualized)
        sharpe_ratio = df['returns'].mean() / df['returns'].std() * np.sqrt(252)
        
        # Calculate drawdown
        df['drawdown'] = (df['total_equity'] / df['total_equity'].cummax() - 1) * 100
        max_drawdown = df['drawdown'].min()
        
        # Calculate win rate
        trades = df[df['position_size'] != 0]
        winning_trades = trades[trades['returns'] > 0]
        win_rate = len(winning_trades) / len(trades) * 100 if len(trades) > 0 else 0
        
        performance = {
            'initial_equity': initial_equity,
            'final_equity': final_equity,
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': len(trades),
        }
        
        return performance
    
    def plot_results(self):
        """Plot equity curve and drawdown."""
        if self.results is None:
            logger.warning("No backtest results to plot.")
            return
        
        df = self.results
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # Plot equity curve
        ax1.plot(df.index, df['total_equity'])
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.set_title('Backtest Results: Equity Curve')
        ax1.grid(True)
        
        # Plot drawdown
        ax2.fill_between(df.index, df['drawdown'], 0, color='red', alpha=0.3)
        ax2.set_ylabel('Drawdown (%)')
        ax2.set_xlabel('Date')
        ax2.set_title('Portfolio Drawdown')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('backtest_results.png')
        plt.close()
        
        logger.info("Backtest results plotted and saved to 'backtest_results.png'")