# utils/backtester.py - Streamlined backtesting framework
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from datetime import datetime
import os

logger = logging.getLogger(__name__)

class Backtester:
    """Class for backtesting trading strategies."""
    
    def __init__(self, initial_capital=10000.0, commission=0.001):
        """
        Initialize backtester.
        
        Args:
            initial_capital: Starting capital for backtest
            commission: Commission rate for trades
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.results = None
    
    def run(self, signals):
        """
        Run backtest on the signals dataframe.
        
        Args:
            signals: DataFrame with price data and trading signals
            
        Returns:
            Dictionary with performance metrics
        """
        logger.info(f"Running backtest with initial capital: ${self.initial_capital}")
        
        # Make a copy of the signals dataframe
        df = signals.copy()
        
        # Check required columns
        required_columns = ['close', 'position']
        if not all(col in df.columns for col in required_columns):
            logger.error(f"Missing required columns: {[col for col in required_columns if col not in df.columns]}")
            return {}
        
        # Initialize portfolio metrics
        df['holdings'] = 0.0
        df['cash'] = self.initial_capital
        df['trade_value'] = 0.0
        df['trade_cost'] = 0.0
        
        # Calculate positions and portfolio values
        position = 0  # Current position
        
        for i in range(len(df)):
            # Get current row
            row = df.iloc[i]
            
            # Calculate trade value and cost if position changes
            if row['position'] != 0:
                position_change = row['position']
                
                # Calculate trade value (positive for buy, negative for sell)
                trade_value = position_change * row['close']
                
                # Calculate commission cost (always positive)
                trade_cost = abs(trade_value) * self.commission
                
                # Update holdings and cash
                df.loc[df.index[i], 'trade_value'] = trade_value
                df.loc[df.index[i], 'trade_cost'] = trade_cost
                
                # Update position
                position += position_change
                
                # Update cash (subtract trade value and cost)
                if i > 0:
                    df.loc[df.index[i], 'cash'] = df.iloc[i-1]['cash'] - trade_value - trade_cost
                else:
                    df.loc[df.index[i], 'cash'] = self.initial_capital - trade_value - trade_cost
            else:
                # No trade
                if i > 0:
                    df.loc[df.index[i], 'cash'] = df.iloc[i-1]['cash']
                else:
                    df.loc[df.index[i], 'cash'] = self.initial_capital
            
            # Update holdings value
            df.loc[df.index[i], 'holdings'] = position * row['close']
        
        # Calculate total equity and returns
        df['total_equity'] = df['cash'] + df['holdings']
        df['returns'] = df['total_equity'].pct_change()
        
        # Calculate drawdown
        df['drawdown'] = (df['total_equity'] / df['total_equity'].cummax() - 1) * 100
        
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
        
        # Annualized return
        days = (df.index[-1] - df.index[0]).days
        if days > 0:
            annualized_return = ((1 + total_return/100) ** (365/days) - 1) * 100
        else:
            annualized_return = 0
        
        # Calculate Sharpe ratio (annualized)
        if df['returns'].std() > 0:
            sharpe_ratio = df['returns'].mean() / df['returns'].std() * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        # Calculate drawdown
        max_drawdown = df['drawdown'].min()
        
        # Calculate win rate
        trades = df[df['trade_value'] != 0]
        if len(trades) > 0:
            # Calculate trade P&L (including commission)
            trades['trade_pnl'] = trades['trade_value'] + trades.shift(-1)['holdings'].fillna(0) - trades['holdings'] - trades['trade_cost']
            winning_trades = trades[trades['trade_pnl'] > 0]
            win_rate = len(winning_trades) / len(trades) * 100
            
            # Calculate profit factor
            gross_profit = winning_trades['trade_pnl'].sum()
            losing_trades = trades[trades['trade_pnl'] <= 0]
            gross_loss = abs(losing_trades['trade_pnl'].sum())
            profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')
        else:
            win_rate = 0
            profit_factor = 0
        
        performance = {
            'initial_equity': initial_equity,
            'final_equity': final_equity,
            'total_return': total_return,
            'annualized_return': annualized_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_trades': len(trades),
            'start_date': df.index[0].strftime('%Y-%m-%d'),
            'end_date': df.index[-1].strftime('%Y-%m-%d'),
        }
        
        return performance
    
    def plot_results(self, save_path=None):
        """
        Plot equity curve and drawdown.
        
        Args:
            save_path: Path to save the plot
        """
        if self.results is None:
            logger.warning("No backtest results to plot.")
            return
        
        df = self.results
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # Plot equity curve
        ax1.plot(df.index, df['total_equity'], label='Portfolio Value')
        ax1.axhline(y=self.initial_capital, color='gray', linestyle='--', label='Initial Capital')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.set_title('Backtest Results: Equity Curve')
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
            logger.info(f"Backtest results plotted and saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def get_trade_summary(self):
        """
        Get a summary of all trades.
        
        Returns:
            DataFrame with trade details
        """
        if self.results is None:
            logger.warning("No backtest results to summarize.")
            return pd.DataFrame()
        
        df = self.results
        
        # Find all trades
        trades = df[df['trade_value'] != 0].copy()
        
        if len(trades) == 0:
            logger.warning("No trades found in backtest results.")
            return pd.DataFrame()
        
        # Calculate trade P&L
        trades['trade_pnl'] = trades['trade_value'] + trades.shift(-1)['holdings'].fillna(0) - trades['holdings'] - trades['trade_cost']
        trades['trade_pnl_pct'] = trades['trade_pnl'] / trades['total_equity'] * 100
        
        # Classify trades
        trades['type'] = 'buy'
        trades.loc[trades['trade_value'] < 0, 'type'] = 'sell'
        
        # Create trade summary
        trade_summary = trades[['type', 'close', 'trade_value', 'trade_cost', 'trade_pnl', 'trade_pnl_pct']]
        trade_summary.columns = ['type', 'price', 'value', 'cost', 'pnl', 'pnl_pct']
        
        return trade_summary