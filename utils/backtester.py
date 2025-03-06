# utils/backtester.py - Enhanced backtesting framework with risk management
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from datetime import datetime
import os
from typing import Dict, Any, Optional, List, Tuple, Union

logger = logging.getLogger(__name__)

class Backtester:
    """Class for backtesting trading strategies with improved risk management."""
    
    def __init__(self, initial_capital=10000.0, commission=0.001, 
                 risk_per_trade=0.02, slippage=0.0, max_position_size=None):
        """
        Initialize backtester.
        
        Args:
            initial_capital: Starting capital for backtest
            commission: Commission rate for trades (as a decimal)
            risk_per_trade: Percentage of capital to risk per trade (as a decimal)
            slippage: Slippage as a percentage of price (as a decimal)
            max_position_size: Maximum position size as percentage of capital (as a decimal)
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.risk_per_trade = risk_per_trade
        self.slippage = slippage
        self.max_position_size = max_position_size
        self.results = None
        self.trade_log = []
    
    def _validate_signals(self, signals: pd.DataFrame) -> bool:
        """
        Validate signals DataFrame before backtesting.
        
        Args:
            signals: DataFrame with price data and trading signals
            
        Returns:
            Boolean indicating if signals are valid
        """
        # Check if DataFrame is empty
        if signals.empty:
            logger.error("Signals DataFrame is empty")
            return False
        
        # Check required columns
        required_columns = ['close', 'position']
        missing_columns = [col for col in required_columns if col not in signals.columns]
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return False
        
        # Check for sufficient data points
        if len(signals) < 10:
            logger.warning("Very few data points for backtesting")
        
        # Check for NaN values in required columns
        for col in required_columns:
            if signals[col].isna().any():
                logger.error(f"NaN values found in {col} column")
                return False
        
        return True
    
    def run(self, signals: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Run backtest on the signals dataframe with improved risk management.
        
        Args:
            signals: DataFrame with price data and trading signals
            
        Returns:
            Dictionary with performance metrics or None if validation fails
        """
        logger.info(f"Running backtest with initial capital: ${self.initial_capital}")
        
        # Validate signals
        if not self._validate_signals(signals):
            logger.error("Signals validation failed")
            return {}
        
        # Make a copy of the signals dataframe
        df = signals.copy()
        
        # Initialize portfolio metrics
        df['holdings'] = 0.0
        df['cash'] = self.initial_capital
        df['trade_value'] = 0.0
        df['trade_cost'] = 0.0
        df['slippage_cost'] = 0.0
        df['position_size'] = 0.0
        
        # Calculate positions and portfolio values
        position = 0  # Current position (number of shares/units)
        position_value = 0  # Current value of the position
        entry_price = 0  # Price at which the position was entered
        
        for i in range(len(df)):
            # Get current row
            row = df.iloc[i]
            current_price = row['close']
            
            # Update holdings value before any trades (mark-to-market)
            current_position_value = position * current_price
            if i > 0:
                df.loc[df.index[i], 'cash'] = df.iloc[i-1]['cash']
            else:
                df.loc[df.index[i], 'cash'] = self.initial_capital
                
            df.loc[df.index[i], 'holdings'] = current_position_value
            
            # Calculate trade value and cost if position changes
            if row['position'] != 0:
                position_change = row['position'] / 2  # Convert back from the 2/-2 format
                
                # Skip trades if we don't have enough cash (preventing negative cash)
                if position_change > 0:  # Buying
                    # Calculate trade value before transaction
                    trade_price = current_price * (1 + self.slippage)  # Add slippage for buys
                    
                    # Calculate position size based on risk management
                    available_capital = df.loc[df.index[i], 'cash']
                    
                    if available_capital <= 0:
                        logger.warning(f"Not enough cash for buy at {df.index[i]}. Skipping trade.")
                        df.loc[df.index[i], 'position'] = 0
                        continue
                    
                    # Calculate position size based on risk per trade
                    capital_to_use = available_capital * self.risk_per_trade
                    
                    # Apply max position size constraint if specified
                    if self.max_position_size is not None:
                        max_position = available_capital * self.max_position_size
                        capital_to_use = min(capital_to_use, max_position)
                    
                    # Calculate new position size
                    units_to_buy = capital_to_use / trade_price
                    
                    # Update values
                    trade_value = units_to_buy * trade_price
                    trade_cost = trade_value * self.commission
                    slippage_cost = trade_value * self.slippage
                    
                    # Ensure we're not spending more than we have
                    if trade_value + trade_cost > available_capital:
                        units_to_buy = available_capital / (trade_price * (1 + self.commission))
                        trade_value = units_to_buy * trade_price
                        trade_cost = trade_value * self.commission
                        slippage_cost = trade_value * self.slippage
                    
                    # Update positions
                    position += units_to_buy
                    entry_price = trade_price
                    
                elif position_change < 0:  # Selling
                    if position <= 0:
                        logger.warning(f"No position to sell at {df.index[i]}. Skipping trade.")
                        df.loc[df.index[i], 'position'] = 0
                        continue
                    
                    # Calculate trade price with slippage
                    trade_price = current_price * (1 - self.slippage)  # Subtract slippage for sells
                    
                    # Calculate units to sell (all current position)
                    units_to_sell = position
                    
                    # Update values
                    trade_value = -units_to_sell * trade_price  # Negative for selling
                    trade_cost = abs(trade_value) * self.commission
                    slippage_cost = abs(trade_value) * self.slippage
                    
                    # Update position
                    position = 0
                    
                    # Log the trade result
                    trade_pnl = (trade_price - entry_price) * units_to_sell - trade_cost - slippage_cost
                    trade_pnl_pct = trade_pnl / (entry_price * units_to_sell) * 100 if entry_price > 0 else 0
                    
                    self.trade_log.append({
                        'entry_date': df.index[i-1],
                        'exit_date': df.index[i],
                        'entry_price': entry_price,
                        'exit_price': trade_price,
                        'position_size': units_to_sell,
                        'trade_pnl': trade_pnl,
                        'trade_pnl_pct': trade_pnl_pct
                    })
                
                # Update dataframe with trade information
                df.loc[df.index[i], 'trade_value'] = trade_value
                df.loc[df.index[i], 'trade_cost'] = trade_cost
                df.loc[df.index[i], 'slippage_cost'] = slippage_cost
                df.loc[df.index[i], 'position_size'] = position
                
                # Update cash (subtract trade value and costs)
                df.loc[df.index[i], 'cash'] = df.loc[df.index[i], 'cash'] - trade_value - trade_cost - slippage_cost
            else:
                # No trade, just update position size
                df.loc[df.index[i], 'position_size'] = position
            
            # Re-calculate holdings after potential trade
            df.loc[df.index[i], 'holdings'] = position * current_price
        
        # Calculate total equity and returns
        df['total_equity'] = df['cash'] + df['holdings']
        df['returns'] = df['total_equity'].pct_change().fillna(0)
        
        # Calculate drawdown
        df['drawdown'] = (df['total_equity'] / df['total_equity'].cummax() - 1) * 100
        
        # Store results
        self.results = df
        
        # Calculate performance metrics
        performance = self._calculate_performance()
        
        logger.info(f"Backtest completed. Final equity: ${performance['final_equity']:.2f}")
        
        return performance
    
    def _calculate_performance(self) -> Dict[str, Any]:
        """
        Calculate comprehensive performance metrics.
        
        Returns:
            Dictionary with performance metrics
        """
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
        
        # Calculate Sharpe ratio (annualized) with risk-free rate
        risk_free_rate = 0.02  # Assume 2% annual risk-free rate
        daily_risk_free = (1 + risk_free_rate) ** (1/365) - 1
        
        if df['returns'].std() > 0:
            excess_returns = df['returns'] - daily_risk_free
            sharpe_ratio = excess_returns.mean() / df['returns'].std() * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        # Calculate Sortino ratio (using only negative returns)
        negative_returns = df['returns'][df['returns'] < 0]
        if len(negative_returns) > 0 and negative_returns.std() > 0:
            sortino_ratio = (df['returns'].mean() - daily_risk_free) / negative_returns.std() * np.sqrt(252)
        else:
            sortino_ratio = 0
        
        # Calculate drawdown statistics
        max_drawdown = df['drawdown'].min()
        avg_drawdown = df['drawdown'][df['drawdown'] < 0].mean() if len(df['drawdown'][df['drawdown'] < 0]) > 0 else 0
        
        # Calculate trade statistics
        trades = df[df['trade_cost'] > 0]
        total_trades = len(trades)
        total_costs = trades['trade_cost'].sum() + trades['slippage_cost'].sum()
        
        # Calculate win rate and other trade metrics from trade log
        if self.trade_log:
            trade_log_df = pd.DataFrame(self.trade_log)
            winning_trades = trade_log_df[trade_log_df['trade_pnl'] > 0]
            losing_trades = trade_log_df[trade_log_df['trade_pnl'] <= 0]
            
            win_rate = len(winning_trades) / len(trade_log_df) * 100 if len(trade_log_df) > 0 else 0
            
            avg_win = winning_trades['trade_pnl'].mean() if len(winning_trades) > 0 else 0
            avg_loss = losing_trades['trade_pnl'].mean() if len(losing_trades) > 0 else 0
            
            # Calculate profit factor
            gross_profit = winning_trades['trade_pnl'].sum() if len(winning_trades) > 0 else 0
            gross_loss = abs(losing_trades['trade_pnl'].sum()) if len(losing_trades) > 0 else 0
            profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')
            
            # Calculate average trade metrics
            avg_trade = trade_log_df['trade_pnl'].mean()
            avg_trade_pct = trade_log_df['trade_pnl_pct'].mean()
            
            # Calculate max consecutive wins/losses
            if len(trade_log_df) > 0:
                trade_log_df['win'] = trade_log_df['trade_pnl'] > 0
                
                # Calculate consecutive wins/losses
                trade_log_df['streak'] = (trade_log_df['win'] != trade_log_df['win'].shift(1)).cumsum()
                winning_streaks = trade_log_df[trade_log_df['win']].groupby('streak').size()
                losing_streaks = trade_log_df[~trade_log_df['win']].groupby('streak').size()
                
                max_consecutive_wins = winning_streaks.max() if len(winning_streaks) > 0 else 0
                max_consecutive_losses = losing_streaks.max() if len(losing_streaks) > 0 else 0
            else:
                max_consecutive_wins = 0
                max_consecutive_losses = 0
        else:
            win_rate = 0
            profit_factor = 0
            avg_win = 0
            avg_loss = 0
            avg_trade = 0
            avg_trade_pct = 0
            max_consecutive_wins = 0
            max_consecutive_losses = 0
        
        # Calculate volatility and risk metrics
        volatility = df['returns'].std() * np.sqrt(252) * 100  # Annualized, in percent
        
        # Calculate Value at Risk (VaR) and Conditional VaR (CVaR/Expected Shortfall)
        var_95 = np.percentile(df['returns'], 5) * 100  # 95% VaR (in percent)
        cvar_95 = df['returns'][df['returns'] <= np.percentile(df['returns'], 5)].mean() * 100  # 95% CVaR (in percent)
        
        # Calculate maximum equity and drawdown duration
        if len(df) > 0:
            max_equity = df['total_equity'].max()
            
            # Calculate drawdown duration
            is_drawdown = df['total_equity'] < df['total_equity'].cummax()
            if is_drawdown.any():
                drawdown_periods = []
                current_period = 0
                
                for i in range(len(df)):
                    if is_drawdown.iloc[i]:
                        current_period += 1
                    elif current_period > 0:
                        drawdown_periods.append(current_period)
                        current_period = 0
                
                if current_period > 0:
                    drawdown_periods.append(current_period)
                
                max_drawdown_duration = max(drawdown_periods) if drawdown_periods else 0
                avg_drawdown_duration = np.mean(drawdown_periods) if drawdown_periods else 0
            else:
                max_drawdown_duration = 0
                avg_drawdown_duration = 0
        else:
            max_equity = initial_equity
            max_drawdown_duration = 0
            avg_drawdown_duration = 0
        
        performance = {
            'initial_equity': initial_equity,
            'final_equity': final_equity,
            'max_equity': max_equity,
            'total_return': total_return,
            'annualized_return': annualized_return,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'volatility': volatility,
            'max_drawdown': max_drawdown,
            'avg_drawdown': avg_drawdown,
            'max_drawdown_duration': max_drawdown_duration,
            'avg_drawdown_duration': avg_drawdown_duration,
            'value_at_risk_95': var_95,
            'cvar_95': cvar_95,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'avg_trade': avg_trade,
            'avg_trade_pct': avg_trade_pct,
            'max_consecutive_wins': max_consecutive_wins,
            'max_consecutive_losses': max_consecutive_losses,
            'total_trades': total_trades,
            'total_costs': total_costs,
            'start_date': df.index[0].strftime('%Y-%m-%d'),
            'end_date': df.index[-1].strftime('%Y-%m-%d'),
            'days': days
        }
        
        return performance
    
    def plot_results(self, save_path=None, show_plot=True):
        """
        Plot comprehensive backtest results.
        
        Args:
            save_path: Path to save the plot
            show_plot: Whether to display the plot
        """
        if self.results is None:
            logger.warning("No backtest results to plot.")
            return
        
        df = self.results
        
        # Create figure with subplots
        fig = plt.figure(figsize=(15, 12))
        gs = fig.add_gridspec(4, 1, height_ratios=[3, 1, 1, 1], hspace=0.3)
        
        # Equity curve
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(df.index, df['total_equity'], label='Portfolio Value')
        ax1.axhline(y=self.initial_capital, color='gray', linestyle='--', label='Initial Capital')
        
        # Add buy/sell markers
        buys = df[df['trade_value'] > 0]
        sells = df[df['trade_value'] < 0]
        
        if not buys.empty:
            ax1.scatter(buys.index, buys['total_equity'], color='green', marker='^', s=50, label='Buy')
        if not sells.empty:
            ax1.scatter(sells.index, sells['total_equity'], color='red', marker='v', s=50, label='Sell')
        
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.set_title('Backtest Results: Equity Curve')
        ax1.legend()
        ax1.grid(True)
        
        # Drawdown
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        ax2.fill_between(df.index, df['drawdown'], 0, color='red', alpha=0.3, label='Drawdown (%)')
        ax2.set_ylabel('Drawdown (%)')
        ax2.set_ylim(bottom=min(df['drawdown'].min() * 1.1, -0.5), top=0.5)
        ax2.legend()
        ax2.grid(True)
        
        # Daily returns
        ax3 = fig.add_subplot(gs[2], sharex=ax1)
        ax3.bar(df.index, df['returns'] * 100, color='blue', alpha=0.7, label='Daily Returns (%)')
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax3.set_ylabel('Return (%)')
        ax3.legend()
        ax3.grid(True)
        
        # Position size
        ax4 = fig.add_subplot(gs[3], sharex=ax1)
        ax4.plot(df.index, df['position_size'], color='purple', label='Position Size')
        ax4.set_ylabel('Position Size')
        ax4.legend()
        ax4.grid(True)
        ax4.set_xlabel('Date')
        
        # Format x-axis dates
        for ax in [ax1, ax2, ax3, ax4]:
            ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        # Save or show the plot
        if save_path:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            logger.info(f"Backtest results plotted and saved to {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def get_trade_summary(self) -> pd.DataFrame:
        """
        Get a detailed summary of all trades.
        
        Returns:
            DataFrame with trade details
        """
        if not self.trade_log:
            logger.warning("No trades found in backtest results.")
            return pd.DataFrame()
        
        # Convert trade log to DataFrame
        trades_df = pd.DataFrame(self.trade_log)
        
        # Add holding period
        trades_df['holding_period'] = (trades_df['exit_date'] - trades_df['entry_date']).dt.days
        
        # Add additional metrics if needed
        trades_df['profit_loss'] = trades_df['trade_pnl']
        trades_df['return_pct'] = trades_df['trade_pnl_pct']
        trades_df['result'] = trades_df['profit_loss'].apply(lambda x: 'Win' if x > 0 else 'Loss')
        
        # Calculate annualized return for each trade
        trades_df['annualized_return'] = trades_df.apply(
            lambda row: ((1 + row['return_pct']/100) ** (365/max(1, row['holding_period'])) - 1) * 100 
            if row['holding_period'] > 0 else 0, 
            axis=1
        )
        
        return trades_df
    
    def monte_carlo_analysis(self, iterations=1000, confidence_level=0.95) -> Dict[str, Any]:
        """
        Perform Monte Carlo analysis on the backtest results.
        
        Args:
            iterations: Number of Monte Carlo iterations
            confidence_level: Confidence level for results
            
        Returns:
            Dictionary with Monte Carlo analysis results
        """
        if self.results is None or len(self.results) < 30:
            logger.warning("Insufficient data for Monte Carlo analysis.")
            return {}
        
        if not self.trade_log:
            logger.warning("No trades found for Monte Carlo analysis.")
            return {}
        
        try:
            # Convert trade log to DataFrame
            trades_df = pd.DataFrame(self.trade_log)
            
            # Extract returns for each trade
            returns = trades_df['trade_pnl_pct'].values / 100  # Convert to decimal
            
            # Generate random samples of returns
            np.random.seed(42)  # For reproducibility
            
            # Store final equity for each simulation
            final_equities = []
            max_drawdowns = []
            
            # Run simulations
            for _ in range(iterations):
                # Shuffle returns
                sampled_returns = np.random.choice(returns, size=len(returns), replace=True)
                
                # Calculate equity curve
                equity_curve = self.initial_capital * np.cumprod(1 + sampled_returns)
                
                # Calculate drawdowns
                cummax = np.maximum.accumulate(equity_curve)
                drawdown = (equity_curve / cummax - 1) * 100
                
                # Store results
                final_equities.append(equity_curve[-1])
                max_drawdowns.append(np.min(drawdown))
            
            # Calculate statistics
            final_equities = np.array(final_equities)
            max_drawdowns = np.array(max_drawdowns)
            
            # Confidence intervals
            alpha = 1 - confidence_level
            lower_idx = int(iterations * alpha / 2)
            upper_idx = int(iterations * (1 - alpha / 2))
            
            sorted_equities = np.sort(final_equities)
            sorted_drawdowns = np.sort(max_drawdowns)
            
            # Calculate percentiles
            equity_05 = np.percentile(final_equities, 5)
            equity_25 = np.percentile(final_equities, 25)
            equity_50 = np.percentile(final_equities, 50)
            equity_75 = np.percentile(final_equities, 75)
            equity_95 = np.percentile(final_equities, 95)
            
            # Probability of loss
            prob_loss = (final_equities < self.initial_capital).mean() * 100
            
            return {
                'iterations': iterations,
                'confidence_level': confidence_level,
                'median_final_equity': equity_50,
                'mean_final_equity': np.mean(final_equities),
                'std_final_equity': np.std(final_equities),
                'equity_05': equity_05,
                'equity_25': equity_25,
                'equity_75': equity_75,
                'equity_95': equity_95,
                'median_max_drawdown': np.median(max_drawdowns),
                'mean_max_drawdown': np.mean(max_drawdowns),
                'worst_max_drawdown': np.min(max_drawdowns),
                'prob_loss': prob_loss,
                'prob_profit': 100 - prob_loss
            }
        except Exception as e:
            logger.error(f"Error in Monte Carlo analysis: {e}")
            return {}