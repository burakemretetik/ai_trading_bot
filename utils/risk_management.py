# utils/risk_management.py - Advanced risk management techniques
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import os

logger = logging.getLogger(__name__)

class PositionSizer:
    """Class for advanced position sizing strategies."""
    
    def __init__(self, initial_capital: float = 10000.0):
        """
        Initialize position sizer.
        
        Args:
            initial_capital: Initial capital
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.open_positions = {}  # Symbol -> (entry_price, size, stop_loss)
    
    def fixed_dollar(self, 
                  available_capital: float,
                  risk_amount: float,
                  entry_price: float) -> float:
        """
        Calculate position size using fixed dollar risk.
        
        Args:
            available_capital: Available capital
            risk_amount: Fixed dollar amount to risk
            entry_price: Entry price
            
        Returns:
            Position size in base currency
        """
        # Ensure risk amount doesn't exceed available capital
        risk_amount = min(risk_amount, available_capital * 0.5)
        
        # Calculate position size based on risk amount
        position_size = risk_amount / entry_price
        
        logger.info(f"Fixed dollar position sizing: ${risk_amount:.2f} at ${entry_price:.2f} = {position_size:.6f} units")
        
        return position_size
    
    def fixed_percent(self, 
                   available_capital: float,
                   risk_percent: float,
                   entry_price: float) -> float:
        """
        Calculate position size using fixed percentage risk.
        
        Args:
            available_capital: Available capital
            risk_percent: Percentage of capital to risk (0-1)
            entry_price: Entry price
            
        Returns:
            Position size in base currency
        """
        # Calculate risk amount
        risk_amount = available_capital * risk_percent
        
        # Calculate position size
        position_size = risk_amount / entry_price
        
        logger.info(f"Fixed percent position sizing: {risk_percent:.1%} of ${available_capital:.2f} = {position_size:.6f} units")
        
        return position_size
    
    def fixed_risk(self, 
                available_capital: float,
                risk_percent: float,
                entry_price: float,
                stop_loss_price: float) -> float:
        """
        Calculate position size using fixed risk (risk per trade).
        
        Args:
            available_capital: Available capital
            risk_percent: Percentage of capital to risk (0-1)
            entry_price: Entry price
            stop_loss_price: Stop loss price
            
        Returns:
            Position size in base currency
        """
        # Calculate risk amount
        risk_amount = available_capital * risk_percent
        
        # Calculate price risk
        price_risk = abs(entry_price - stop_loss_price)
        
        # Check for invalid price risk
        if price_risk == 0:
            logger.warning("Stop loss price equals entry price, using default 1% price risk")
            price_risk = entry_price * 0.01
        
        # Calculate position size
        position_size = risk_amount / price_risk
        
        logger.info(f"Fixed risk position sizing: {risk_percent:.1%} of ${available_capital:.2f} "
                 f"with price risk ${price_risk:.2f} = {position_size:.6f} units")
        
        return position_size
    
    def volatility_based(self, 
                      available_capital: float,
                      risk_factor: float,
                      entry_price: float,
                      volatility: float) -> float:
        """
        Calculate position size based on volatility.
        
        Args:
            available_capital: Available capital
            risk_factor: Risk factor (lower = more conservative)
            entry_price: Entry price
            volatility: Volatility measure (e.g., ATR)
            
        Returns:
            Position size in base currency
        """
        # Check for invalid volatility
        if volatility <= 0:
            logger.warning("Invalid volatility value, using default 1% volatility")
            volatility = entry_price * 0.01
        
        # Calculate risk per share
        risk_per_unit = volatility * risk_factor
        
        # Calculate max risk amount (e.g., 2% of capital)
        max_risk_amount = available_capital * 0.02
        
        # Calculate position size
        position_size = max_risk_amount / risk_per_unit
        
        logger.info(f"Volatility-based position sizing: "
                 f"Volatility ${volatility:.2f}, Risk Factor {risk_factor:.2f}, "
                 f"Risk per unit ${risk_per_unit:.2f} = {position_size:.6f} units")
        
        return position_size
    
    def kelly_criterion(self, 
                     available_capital: float,
                     win_rate: float,
                     win_loss_ratio: float,
                     entry_price: float,
                     fraction: float = 1.0) -> float:
        """
        Calculate position size using Kelly Criterion.
        
        Args:
            available_capital: Available capital
            win_rate: Historical win rate (0-1)
            win_loss_ratio: Ratio of average win to average loss
            entry_price: Entry price
            fraction: Fraction of Kelly to use (0-1, lower = more conservative)
            
        Returns:
            Position size in base currency
        """
        # Calculate Kelly percentage
        kelly_pct = win_rate - (1 - win_rate) / win_loss_ratio
        
        # Apply fraction and ensure non-negative
        kelly_pct = max(0, kelly_pct * fraction)
        
        # Cap at 25% to avoid excessive risk
        kelly_pct = min(kelly_pct, 0.25)
        
        # Calculate position size
        position_value = available_capital * kelly_pct
        position_size = position_value / entry_price
        
        logger.info(f"Kelly Criterion position sizing: "
                 f"Win Rate {win_rate:.2f}, Win/Loss Ratio {win_loss_ratio:.2f}, "
                 f"Kelly % {kelly_pct:.2%} = {position_size:.6f} units")
        
        return position_size
    
    def update_capital(self, current_capital: float) -> None:
        """
        Update current capital.
        
        Args:
            current_capital: Current capital amount
        """
        self.current_capital = current_capital
    
    def pyramiding(self, 
                available_capital: float,
                risk_percent: float,
                entry_price: float,
                stop_loss_price: float,
                target_price: float,
                current_price: float,
                existing_position: float,
                max_entries: int = 3) -> float:
        """
        Calculate position size for pyramiding (adding to position).
        
        Args:
            available_capital: Available capital
            risk_percent: Percentage of capital to risk per trade (0-1)
            entry_price: Entry price for new position
            stop_loss_price: Stop loss price
            target_price: Target price
            current_price: Current market price
            existing_position: Existing position size
            max_entries: Maximum number of entries
            
        Returns:
            Additional position size in base currency
        """
        # Check if we already have too many entries
        if len(self.open_positions) >= max_entries:
            logger.info(f"Maximum number of entries ({max_entries}) reached, no additional position")
            return 0.0
        
        # Calculate price risk
        price_risk = abs(entry_price - stop_loss_price)
        
        # Check risk/reward ratio (at least 2:1)
        reward = abs(target_price - entry_price)
        risk_reward_ratio = reward / price_risk
        
        if risk_reward_ratio < 2.0:
            logger.info(f"Risk/reward ratio ({risk_reward_ratio:.2f}) too low for pyramiding")
            return 0.0
        
        # Calculate basic position size
        risk_amount = available_capital * risk_percent
        position_size = risk_amount / price_risk
        
        # Reduce size for subsequent entries
        position_size = position_size / (len(self.open_positions) + 1)
        
        logger.info(f"Pyramiding position sizing: Entry #{len(self.open_positions) + 1}, "
                 f"Size: {position_size:.6f} units")
        
        return position_size
    
    def position_size_with_correlation(self, 
                                    available_capital: float,
                                    risk_percent: float,
                                    entry_price: float,
                                    stop_loss_price: float,
                                    correlation: float) -> float:
        """
        Calculate position size adjusting for correlation with existing positions.
        
        Args:
            available_capital: Available capital
            risk_percent: Percentage of capital to risk (0-1)
            entry_price: Entry price
            stop_loss_price: Stop loss price
            correlation: Correlation with existing portfolio (-1 to 1)
            
        Returns:
            Position size in base currency
        """
        # Calculate risk amount
        risk_amount = available_capital * risk_percent
        
        # Calculate price risk
        price_risk = abs(entry_price - stop_loss_price)
        
        # Check for invalid price risk
        if price_risk == 0:
            logger.warning("Stop loss price equals entry price, using default 1% price risk")
            price_risk = entry_price * 0.01
        
        # Calculate base position size
        base_position_size = risk_amount / price_risk
        
        # Adjust for correlation (higher correlation = smaller position)
        adjustment_factor = 1.0 - abs(max(0, correlation))
        
        # Calculate adjusted position size
        position_size = base_position_size * adjustment_factor
        
        logger.info(f"Correlation-adjusted position sizing: Base size {base_position_size:.6f}, "
                 f"Correlation {correlation:.2f}, Adjustment {adjustment_factor:.2f}, "
                 f"Final size {position_size:.6f} units")
        
        return position_size


class StopLossManager:
    """Class for advanced stop loss management techniques."""
    
    def __init__(self):
        """Initialize stop loss manager."""
        pass
    
    def fixed_price(self, 
                 entry_price: float,
                 risk_percent: float,
                 direction: str = 'long') -> float:
        """
        Calculate fixed price stop loss.
        
        Args:
            entry_price: Entry price
            risk_percent: Percentage of entry price to risk (0-1)
            direction: Trade direction ('long' or 'short')
            
        Returns:
            Stop loss price
        """
        if direction.lower() == 'long':
            stop_loss = entry_price * (1 - risk_percent)
        else:  # short
            stop_loss = entry_price * (1 + risk_percent)
        
        logger.info(f"Fixed price stop loss: {direction}, Entry ${entry_price:.2f}, "
                 f"Risk {risk_percent:.1%}, Stop ${stop_loss:.2f}")
        
        return stop_loss
    
    def volatility_based(self, 
                      entry_price: float,
                      atr: float,
                      multiplier: float = 2.0,
                      direction: str = 'long') -> float:
        """
        Calculate volatility-based stop loss using ATR.
        
        Args:
            entry_price: Entry price
            atr: Average True Range value
            multiplier: ATR multiplier
            direction: Trade direction ('long' or 'short')
            
        Returns:
            Stop loss price
        """
        if direction.lower() == 'long':
            stop_loss = entry_price - (atr * multiplier)
        else:  # short
            stop_loss = entry_price + (atr * multiplier)
        
        logger.info(f"Volatility-based stop loss: {direction}, Entry ${entry_price:.2f}, "
                 f"ATR ${atr:.2f}, Multiplier {multiplier:.1f}, Stop ${stop_loss:.2f}")
        
        return stop_loss
    
    def trailing_stop(self, 
                   current_price: float,
                   highest_price: float,
                   lowest_price: float,
                   trail_percent: float,
                   direction: str = 'long') -> float:
        """
        Calculate trailing stop loss.
        
        Args:
            current_price: Current market price
            highest_price: Highest price since entry
            lowest_price: Lowest price since entry
            trail_percent: Trailing percentage (0-1)
            direction: Trade direction ('long' or 'short')
            
        Returns:
            Stop loss price
        """
        if direction.lower() == 'long':
            stop_loss = highest_price * (1 - trail_percent)
        else:  # short
            stop_loss = lowest_price * (1 + trail_percent)
        
        logger.info(f"Trailing stop loss: {direction}, Current ${current_price:.2f}, "
                 f"{'Highest' if direction.lower() == 'long' else 'Lowest'} "
                 f"${highest_price if direction.lower() == 'long' else lowest_price:.2f}, "
                 f"Trail {trail_percent:.1%}, Stop ${stop_loss:.2f}")
        
        return stop_loss
    
    def chandelier_exit(self, 
                     high_price: float,
                     low_price: float,
                     atr: float,
                     multiplier: float = 3.0,
                     direction: str = 'long') -> float:
        """
        Calculate Chandelier Exit stop loss.
        
        Args:
            high_price: Highest price since entry
            low_price: Lowest price since entry
            atr: Average True Range value
            multiplier: ATR multiplier
            direction: Trade direction ('long' or 'short')
            
        Returns:
            Stop loss price
        """
        if direction.lower() == 'long':
            stop_loss = high_price - (atr * multiplier)
        else:  # short
            stop_loss = low_price + (atr * multiplier)
        
        logger.info(f"Chandelier Exit stop loss: {direction}, "
                 f"{'High' if direction.lower() == 'long' else 'Low'} "
                 f"${high_price if direction.lower() == 'long' else low_price:.2f}, "
                 f"ATR ${atr:.2f}, Multiplier {multiplier:.1f}, Stop ${stop_loss:.2f}")
        
        return stop_loss
    
    def time_based(self, 
                price_data: pd.DataFrame,
                entry_time: datetime,
                max_bars: int = 5,
                direction: str = 'long') -> bool:
        """
        Determine whether to exit a trade based on time (bars).
        
        Args:
            price_data: DataFrame with price data
            entry_time: Entry time
            max_bars: Maximum number of bars to hold
            direction: Trade direction ('long' or 'short')
            
        Returns:
            True if time-based exit should be triggered
        """
        # Find entry index
        if entry_time not in price_data.index:
            logger.warning(f"Entry time {entry_time} not found in price data")
            return False
        
        entry_idx = price_data.index.get_loc(entry_time)
        current_idx = len(price_data) - 1
        
        # Calculate bars since entry
        bars_since_entry = current_idx - entry_idx
        
        # Determine if we should exit
        should_exit = bars_since_entry >= max_bars
        
        if should_exit:
            logger.info(f"Time-based stop triggered: {direction}, "
                     f"Bars since entry {bars_since_entry}, Max bars {max_bars}")
        
        return should_exit
    
    def parabolic_sar(self, 
                   price_data: pd.DataFrame,
                   af_start: float = 0.02,
                   af_increment: float = 0.02,
                   af_max: float = 0.2,
                   direction: str = 'long') -> pd.Series:
        """
        Calculate Parabolic SAR stop loss.
        
        Args:
            price_data: DataFrame with price data
            af_start: Starting acceleration factor
            af_increment: Acceleration factor increment
            af_max: Maximum acceleration factor
            direction: Trade direction ('long' or 'short')
            
        Returns:
            Series with Parabolic SAR values
        """
        high = price_data['high']
        low = price_data['low']
        
        # Prepare SAR series
        sar = pd.Series(index=price_data.index)
        
        # Initial values
        if direction.lower() == 'long':
            # Start with the lowest point for longs
            sar.iloc[0] = low.iloc[0]
            ep = high.iloc[0]  # Extreme point
        else:  # short
            # Start with the highest point for shorts
            sar.iloc[0] = high.iloc[0]
            ep = low.iloc[0]  # Extreme point
        
        af = af_start  # Acceleration factor
        
        # Calculate SAR values
        for i in range(1, len(sar)):
            # Calculate SAR
            sar.iloc[i] = sar.iloc[i-1] + af * (ep - sar.iloc[i-1])
            
            # Update extreme point and acceleration factor
            if direction.lower() == 'long':
                if high.iloc[i] > ep:
                    ep = high.iloc[i]
                    af = min(af + af_increment, af_max)
                
                # Ensure SAR is below the low of the previous periods
                sar.iloc[i] = min(sar.iloc[i], low.iloc[max(0, i-2):i].min())
            else:  # short
                if low.iloc[i] < ep:
                    ep = low.iloc[i]
                    af = min(af + af_increment, af_max)
                
                # Ensure SAR is above the high of the previous periods
                sar.iloc[i] = max(sar.iloc[i], high.iloc[max(0, i-2):i].max())
        
        return sar
    
    def plot_stop_losses(self, 
                      price_data: pd.DataFrame,
                      stop_losses: Dict[str, pd.Series],
                      entry_price: Optional[float] = None,
                      direction: str = 'long',
                      figsize: Tuple[int, int] = (12, 8),
                      save_path: Optional[str] = None) -> None:
        """
        Plot price data with multiple stop loss methods.
        
        Args:
            price_data: DataFrame with price data
            stop_losses: Dictionary of stop loss method name -> Series
            entry_price: Entry price
            direction: Trade direction ('long' or 'short')
            figsize: Figure size
            save_path: Path to save the plot
        """
        plt.figure(figsize=figsize)
        
        # Plot price
        plt.plot(price_data.index, price_data['close'], label='Close Price', color='blue')
        
        # Plot each stop loss method
        for name, stop_loss in stop_losses.items():
            plt.plot(stop_loss.index, stop_loss, label=f"{name} Stop", linestyle='--')
        
        # Plot entry price if provided
        if entry_price is not None:
            plt.axhline(y=entry_price, color='green', linestyle='-', 
                     label=f"Entry (${entry_price:.2f})")
        
        # Add title and labels
        plt.title(f"{direction.capitalize()} Trade Stop Loss Methods", fontsize=16)
        plt.xlabel('Date', fontsize=14)
        plt.ylabel('Price', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        
        # Save or show the plot
        if save_path:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            logger.info(f"Stop loss plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()


class PortfolioRiskManager:
    """Class for portfolio-level risk management."""
    
    def __init__(self, initial_capital: float = 10000.0):
        """
        Initialize portfolio risk manager.
        
        Args:
            initial_capital: Initial capital
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.open_positions = {}  # Symbol -> (entry_price, size, stop_loss)
        self.max_correlated_positions = 3
        self.max_sector_exposure = 0.25  # 25% max exposure to any sector
        self.max_drawdown_limit = 0.15  # 15% max drawdown
        self.max_open_positions = 10
        self.correlation_matrix = None
        self.vola_windows = [10, 21, 63]  # Trading days (2 weeks, 1 month, 3 months)
    
    def update_capital(self, current_capital: float) -> None:
        """
        Update current capital.
        
        Args:
            current_capital: Current capital amount
        """
        self.current_capital = current_capital
    
    def check_drawdown_limit(self, current_drawdown: float) -> bool:
        """
        Check if drawdown has exceeded the limit.
        
        Args:
            current_drawdown: Current drawdown as a decimal
            
        Returns:
            True if drawdown is within limit
        """
        within_limit = abs(current_drawdown) < self.max_drawdown_limit
        
        if not within_limit:
            logger.warning(f"Drawdown limit exceeded: {current_drawdown:.2%} > {self.max_drawdown_limit:.2%}")
        
        return within_limit
    
    def position_exists(self, symbol: str) -> bool:
        """
        Check if position exists for symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            True if position exists
        """
        return symbol in self.open_positions
    
    def add_position(self, 
                  symbol: str,
                  entry_price: float,
                  size: float,
                  stop_loss: float) -> bool:
        """
        Add new position.
        
        Args:
            symbol: Trading symbol
            entry_price: Entry price
            size: Position size
            stop_loss: Stop loss price
            
        Returns:
            True if position added successfully
        """
        # Check if we already have too many open positions
        if len(self.open_positions) >= self.max_open_positions:
            logger.warning(f"Max open positions ({self.max_open_positions}) reached, cannot add position for {symbol}")
            return False
        
        # Add position
        self.open_positions[symbol] = (entry_price, size, stop_loss)
        
        logger.info(f"Position added: {symbol}, Entry: ${entry_price:.2f}, Size: {size:.6f}, Stop: ${stop_loss:.2f}")
        
        return True
    
    def remove_position(self, symbol: str) -> bool:
        """
        Remove position.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            True if position removed successfully
        """
        if symbol in self.open_positions:
            del self.open_positions[symbol]
            logger.info(f"Position removed: {symbol}")
            return True
        else:
            logger.warning(f"Position not found: {symbol}")
            return False
    
    def update_stop_loss(self, 
                      symbol: str,
                      new_stop_loss: float) -> bool:
        """
        Update stop loss for existing position.
        
        Args:
            symbol: Trading symbol
            new_stop_loss: New stop loss price
            
        Returns:
            True if stop loss updated successfully
        """
        if symbol in self.open_positions:
            entry_price, size, old_stop_loss = self.open_positions[symbol]
            self.open_positions[symbol] = (entry_price, size, new_stop_loss)
            
            logger.info(f"Stop loss updated for {symbol}: ${old_stop_loss:.2f} -> ${new_stop_loss:.2f}")
            
            return True
        else:
            logger.warning(f"Position not found: {symbol}")
            return False
    
    def calculate_portfolio_risk(self, 
                             current_prices: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate portfolio risk metrics.
        
        Args:
            current_prices: Dictionary of symbol -> current price
            
        Returns:
            Dictionary of risk metrics
        """
        # Initialize risk metrics
        risk_metrics = {
            'total_exposure': 0.0,
            'total_risk': 0.0,
            'largest_position_pct': 0.0,
            'average_risk_per_trade': 0.0,
            'potential_loss': 0.0
        }
        
        # Calculate metrics
        total_exposure = 0.0
        total_risk = 0.0
        largest_position = 0.0
        
        for symbol, (entry_price, size, stop_loss) in self.open_positions.items():
            # Get current price
            current_price = current_prices.get(symbol, entry_price)
            
            # Calculate position value
            position_value = size * current_price
            total_exposure += position_value
            
            # Calculate risk
            risk_per_unit = abs(current_price - stop_loss)
            position_risk = size * risk_per_unit
            total_risk += position_risk
            
            # Track largest position
            largest_position = max(largest_position, position_value)
        
        # Calculate portfolio metrics
        if self.open_positions:
            risk_metrics['total_exposure'] = total_exposure
            risk_metrics['total_exposure_pct'] = total_exposure / self.current_capital
            risk_metrics['total_risk'] = total_risk
            risk_metrics['total_risk_pct'] = total_risk / self.current_capital
            risk_metrics['largest_position_pct'] = largest_position / self.current_capital
            risk_metrics['average_risk_per_trade'] = total_risk / len(self.open_positions)
            risk_metrics['potential_loss_pct'] = total_risk / self.current_capital
            risk_metrics['num_positions'] = len(self.open_positions)
        
        return risk_metrics
    
    def calculate_position_sizing(self, 
                               symbol: str,
                               entry_price: float,
                               stop_loss_price: float,
                               volatility: float,
                               risk_per_trade: float = 0.01) -> float:
        """
        Calculate appropriate position size based on portfolio risk.
        
        Args:
            symbol: Trading symbol
            entry_price: Entry price
            stop_loss_price: Stop loss price
            volatility: Volatility measure
            risk_per_trade: Risk per trade as percentage of capital
            
        Returns:
            Recommended position size
        """
        # Calculate available capital
        available_capital = self.current_capital
        
        # Get current portfolio risk metrics
        risk_metrics = self.calculate_portfolio_risk({})
        current_risk_pct = risk_metrics.get('total_risk_pct', 0.0)
        
        # Adjust risk per trade if portfolio risk is already high
        if current_risk_pct > 0.05:  # If already risking more than 5%
            adjustment_factor = 1.0 - (current_risk_pct / 0.1)  # Reduce gradually as we approach 10%
            adjustment_factor = max(0.2, adjustment_factor)  # Don't go below 20% of original risk
            risk_per_trade *= adjustment_factor
            
            logger.info(f"Adjusting risk per trade due to high portfolio risk: {risk_per_trade:.2%}")
        
        # Calculate position size
        price_risk = abs(entry_price - stop_loss_price)
        
        # Ensure valid price risk
        if price_risk <= 0:
            price_risk = entry_price * 0.01  # Default 1% price risk
        
        risk_amount = available_capital * risk_per_trade
        position_size = risk_amount / price_risk
        
        # Adjust based on volatility
        if volatility > 0:
            vol_adjustment = 1.0 / (1.0 + volatility)  # Higher volatility = smaller position
            position_size *= vol_adjustment
            
            logger.info(f"Volatility adjustment: {vol_adjustment:.2f}")
        
        # Check number of open positions
        if len(self.open_positions) >= self.max_open_positions / 2:
            # Reduce position size as we approach max positions
            pos_adjustment = 1.0 - (len(self.open_positions) / self.max_open_positions)
            position_size *= pos_adjustment
            
            logger.info(f"Open positions adjustment: {pos_adjustment:.2f}")
        
        logger.info(f"Calculated position size for {symbol}: {position_size:.6f} units")
        
        return position_size
    
    def check_correlation_limit(self, 
                             symbol: str,
                             market_data: Dict[str, pd.DataFrame],
                             threshold: float = 0.7) -> bool:
        """
        Check if adding this position would exceed correlation limits.
        
        Args:
            symbol: Trading symbol
            market_data: Dictionary of symbol -> price DataFrame
            threshold: Correlation threshold
            
        Returns:
            True if correlation is acceptable
        """
        # If we don't have enough open positions, no correlation issue
        if len(self.open_positions) < 2:
            return True
        
        # Calculate correlation matrix if not already calculated
        if self.correlation_matrix is None:
            self._calculate_correlation_matrix(market_data)
        
        # Check correlation with existing positions
        high_correlation_count = 0
        
        for existing_symbol in self.open_positions.keys():
            if symbol == existing_symbol:
                continue
                
            # Get correlation
            if symbol in self.correlation_matrix and existing_symbol in self.correlation_matrix[symbol]:
                correlation = self.correlation_matrix[symbol][existing_symbol]
                
                if abs(correlation) > threshold:
                    high_correlation_count += 1
                    logger.info(f"High correlation between {symbol} and {existing_symbol}: {correlation:.2f}")
        
        # Check if we have too many correlated positions
        if high_correlation_count >= self.max_correlated_positions:
            logger.warning(f"Too many correlated positions for {symbol}: {high_correlation_count} > {self.max_correlated_positions}")
            return False
        
        return True
    
    def _calculate_correlation_matrix(self, market_data: Dict[str, pd.DataFrame]) -> None:
        """
        Calculate correlation matrix for all symbols.
        
        Args:
            market_data: Dictionary of symbol -> price DataFrame
        """
        # Extract close prices
        close_prices = {}
        
        for symbol, data in market_data.items():
            if 'close' in data.columns:
                close_prices[symbol] = data['close']
        
        # Create DataFrame with all close prices
        if close_prices:
            all_prices = pd.DataFrame(close_prices)
            
            # Calculate correlation matrix
            correlation_matrix = all_prices.corr()
            
            # Convert to dictionary for easier access
            self.correlation_matrix = {}
            
            for symbol1 in correlation_matrix.index:
                self.correlation_matrix[symbol1] = {}
                
                for symbol2 in correlation_matrix.columns:
                    self.correlation_matrix[symbol1][symbol2] = correlation_matrix.loc[symbol1, symbol2]
        
        logger.info("Correlation matrix calculated")
    
    def calculate_volatility_rank(self, 
                               market_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """
        Calculate volatility rank for each symbol.
        
        Args:
            market_data: Dictionary of symbol -> price DataFrame
            
        Returns:
            Dictionary of symbol -> volatility rank
        """
        volatility_rank = {}
        
        for symbol, data in market_data.items():
            if 'close' in data.columns and len(data) > max(self.vola_windows):
                # Calculate volatility for each window
                volatilities = []
                
                for window in self.vola_windows:
                    vola = data['close'].pct_change().rolling(window=window).std().iloc[-1]
                    volatilities.append(vola)
                
                # Calculate average volatility
                avg_vola = sum(volatilities) / len(volatilities)
                volatility_rank[symbol] = avg_vola
        
        logger.info(f"Calculated volatility rank for {len(volatility_rank)} symbols")
        
        return volatility_rank
    
    def check_sector_exposure(self, 
                          symbol: str,
                          sector: str,
                          entry_price: float,
                          size: float,
                          sector_mapping: Dict[str, str]) -> bool:
        """
        Check if adding this position would exceed sector exposure limit.
        
        Args:
            symbol: Trading symbol
            sector: Sector
            entry_price: Entry price
            size: Position size
            sector_mapping: Dictionary of symbol -> sector
            
        Returns:
            True if sector exposure is acceptable
        """
        # Calculate current sector exposure
        sector_exposure = {}
        
        for sym, (price, sz, _) in self.open_positions.items():
            sym_sector = sector_mapping.get(sym, 'Unknown')
            
            if sym_sector not in sector_exposure:
                sector_exposure[sym_sector] = 0.0
            
            sector_exposure[sym_sector] += price * sz
        
        # Calculate total exposure
        total_exposure = sum(sector_exposure.values())
        
        # Convert to percentages
        for sec, exposure in sector_exposure.items():
            sector_exposure[sec] = exposure / self.current_capital
        
        # Calculate new exposure
        new_position_value = entry_price * size
        new_sector_exposure = sector_exposure.get(sector, 0.0) + (new_position_value / self.current_capital)
        
        # Check if new exposure exceeds limit
        if new_sector_exposure > self.max_sector_exposure:
            logger.warning(f"Sector exposure limit exceeded for {sector}: {new_sector_exposure:.2%} > {self.max_sector_exposure:.2%}")
            return False
        
        return True
    
    def plot_portfolio_risk(self, 
                        risk_history: pd.DataFrame,
                        figsize: Tuple[int, int] = (12, 8),
                        save_path: Optional[str] = None) -> None:
        """
        Plot portfolio risk metrics over time.
        
        Args:
            risk_history: DataFrame with risk metrics history
            figsize: Figure size
            save_path: Path to save the plot
        """
        plt.figure(figsize=figsize)
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
        
        # Plot exposure and risk
        ax1.plot(risk_history.index, risk_history['total_exposure_pct'] * 100, 
               label='Total Exposure (%)', color='blue')
        ax1.plot(risk_history.index, risk_history['total_risk_pct'] * 100, 
               label='Total Risk (%)', color='red')
        
        # Add horizontal lines for limits
        ax1.axhline(y=50, color='blue', linestyle=':', alpha=0.5, label='50% Exposure')
        ax1.axhline(y=self.max_drawdown_limit * 100, color='red', linestyle=':', 
                  alpha=0.5, label=f"{self.max_drawdown_limit*100:.0f}% Risk")
        
        # Add labels
        ax1.set_title('Portfolio Risk Metrics', fontsize=16)
        ax1.set_ylabel('Percentage of Capital', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot number of positions and largest position
        ax2.plot(risk_history.index, risk_history['num_positions'], 
               label='Open Positions', color='green')
        ax2.plot(risk_history.index, risk_history['largest_position_pct'] * 100, 
               label='Largest Position (%)', color='orange')
        
        # Add horizontal line for max positions
        ax2.axhline(y=self.max_open_positions, color='green', linestyle=':', 
                  alpha=0.5, label=f"{self.max_open_positions} Max Positions")
        
        # Add labels
        ax2.set_ylabel('Count / Percentage', fontsize=14)
        ax2.set_xlabel('Date', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        
        # Save or show the plot
        if save_path:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            logger.info(f"Portfolio risk plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()