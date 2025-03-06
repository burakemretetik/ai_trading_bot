# strategies/lstm_strategy.py - LSTM-based trading strategy
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
import os
import matplotlib.pyplot as plt

from models.sequence_model import TimeSeriesPredictor
from utils.risk_management import PositionSizer, StopLossManager

logger = logging.getLogger(__name__)

class LSTMStrategy:
    """Trading strategy based on LSTM/GRU predictions."""
    
    def __init__(self, 
               threshold: float = 0.002, 
               sequence_length: int = 10,
               hidden_dim: int = 64,
               num_layers: int = 2,
               dropout: float = 0.2,
               cell_type: str = 'lstm',
               use_dynamic_threshold: bool = False,
               use_stop_loss: bool = True,
               stop_loss_pct: float = 0.02,
               use_trailing_stop: bool = False,
               trailing_stop_pct: float = 0.02,
               take_profit_pct: float = 0.05,
               risk_per_trade: float = 0.02,
               use_volatility_sizing: bool = True):
        """
        Initialize LSTM strategy.
        
        Args:
            threshold: Minimum predicted return to enter a position
            sequence_length: Input sequence length for LSTM model
            hidden_dim: Hidden dimension for LSTM model
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            cell_type: Cell type ('lstm' or 'gru')
            use_dynamic_threshold: Whether to use dynamic threshold based on volatility
            use_stop_loss: Whether to use stop loss
            stop_loss_pct: Stop loss percentage
            use_trailing_stop: Whether to use trailing stop
            trailing_stop_pct: Trailing stop percentage
            take_profit_pct: Take profit percentage
            risk_per_trade: Risk per trade as percentage of capital
            use_volatility_sizing: Whether to use volatility-based position sizing
        """
        self.threshold = threshold
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.cell_type = cell_type
        self.use_dynamic_threshold = use_dynamic_threshold
        self.use_stop_loss = use_stop_loss
        self.stop_loss_pct = stop_loss_pct
        self.use_trailing_stop = use_trailing_stop
        self.trailing_stop_pct = trailing_stop_pct
        self.take_profit_pct = take_profit_pct
        self.risk_per_trade = risk_per_trade
        self.use_volatility_sizing = use_volatility_sizing
        
        # Initialize predictor
        self.predictor = TimeSeriesPredictor(
            sequence_length=sequence_length,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            cell_type=cell_type
        )
        
        # Initialize risk management
        self.position_sizer = PositionSizer()
        self.stop_loss_manager = StopLossManager()
        
        # Trading state
        self.last_trained = None
        self.stop_loss_price = None
        self.take_profit_price = None
        self.open_position_price = None
        self.highest_price = None
        self.lowest_price = None
    
    def train_model(self, data: pd.DataFrame) -> None:
        """
        Train the LSTM model on historical data.
        
        Args:
            data: DataFrame with OHLCV data
        """
        logger.info("Training LSTM model for trading strategy...")
        
        # Train model
        results = self.predictor.train(data, verbose=False)
        
        # Log training results
        if results:
            logger.info(f"Model trained. MSE: {results['test_mse']:.6f}, R²: {results['test_r2']:.4f}")
        
        # Store training date
        self.last_trained = data.index[-1]
    
    def calculate_dynamic_threshold(self, data: pd.DataFrame) -> float:
        """
        Calculate dynamic threshold based on volatility.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dynamic threshold value
        """
        # Calculate volatility (10-day standard deviation of returns)
        volatility = data['returns'].rolling(window=10).std().iloc[-1]
        
        # Set threshold as a multiple of volatility
        dynamic_threshold = volatility * 0.5  # Half of volatility
        
        # Ensure minimum threshold
        dynamic_threshold = max(dynamic_threshold, 0.001)
        
        logger.info(f"Dynamic threshold: {dynamic_threshold:.4f} (Volatility: {volatility:.4f})")
        
        return dynamic_threshold
    
    def calculate_position_size(self, 
                             data: pd.DataFrame, 
                             entry_price: float, 
                             capital: float = 10000.0) -> float:
        """
        Calculate position size.
        
        Args:
            data: DataFrame with OHLCV data
            entry_price: Entry price
            capital: Available capital
            
        Returns:
            Position size
        """
        if self.use_volatility_sizing:
            # Calculate ATR for volatility
            high_low = data['high'] - data['low']
            high_close = abs(data['high'] - data['close'].shift(1))
            low_close = abs(data['low'] - data['close'].shift(1))
            
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = true_range.rolling(window=14).mean().iloc[-1]
            
            # Calculate stop loss price based on ATR
            if self.use_stop_loss:
                stop_loss_price = entry_price - (atr * 2)  # 2 ATR stop loss
                self.stop_loss_price = stop_loss_price
            else:
                stop_loss_price = entry_price * (1 - self.stop_loss_pct)
                self.stop_loss_price = stop_loss_price
            
            # Calculate position size based on risk
            size = self.position_sizer.fixed_risk(
                available_capital=capital,
                risk_percent=self.risk_per_trade,
                entry_price=entry_price,
                stop_loss_price=stop_loss_price
            )
        else:
            # Fixed percentage risk
            size = self.position_sizer.fixed_percent(
                available_capital=capital,
                risk_percent=self.risk_per_trade,
                entry_price=entry_price
            )
            
            # Calculate stop loss price
            if self.use_stop_loss:
                self.stop_loss_price = entry_price * (1 - self.stop_loss_pct)
        
        # Calculate take profit price
        self.take_profit_price = entry_price * (1 + self.take_profit_pct)
        
        return size
    
    def update_stops(self, current_price: float, position: int) -> Tuple[Optional[float], bool]:
        """
        Update stop loss and take profit levels.
        
        Args:
            current_price: Current market price
            position: Current position (1=long, -1=short, 0=none)
            
        Returns:
            Tuple of (new_stop_loss_price, should_exit)
        """
        should_exit = False
        new_stop_loss = self.stop_loss_price
        
        # If no position, return default values
        if position == 0 or self.open_position_price is None:
            return None, False
        
        # Update price extremes
        if self.highest_price is None or current_price > self.highest_price:
            self.highest_price = current_price
        
        if self.lowest_price is None or current_price < self.lowest_price:
            self.lowest_price = current_price
        
        # Check if take profit is hit
        if position == 1 and current_price >= self.take_profit_price:  # Long position
            logger.info(f"Take profit hit: {current_price:.2f} >= {self.take_profit_price:.2f}")
            should_exit = True
        elif position == -1 and current_price <= self.take_profit_price:  # Short position
            logger.info(f"Take profit hit: {current_price:.2f} <= {self.take_profit_price:.2f}")
            should_exit = True
        
        # Check if stop loss is hit
        if position == 1 and current_price <= self.stop_loss_price:  # Long position
            logger.info(f"Stop loss hit: {current_price:.2f} <= {self.stop_loss_price:.2f}")
            should_exit = True
        elif position == -1 and current_price >= self.stop_loss_price:  # Short position
            logger.info(f"Stop loss hit: {current_price:.2f} >= {self.stop_loss_price:.2f}")
            should_exit = True
        
        # Update trailing stop if enabled
        if self.use_trailing_stop and not should_exit:
            if position == 1:  # Long position
                # Calculate new trailing stop
                trailing_stop = self.highest_price * (1 - self.trailing_stop_pct)
                
                # Only move stop loss up, never down
                if trailing_stop > self.stop_loss_price:
                    new_stop_loss = trailing_stop
                    logger.info(f"Trailing stop updated: {self.stop_loss_price:.2f} -> {new_stop_loss:.2f}")
                    self.stop_loss_price = new_stop_loss
            
            elif position == -1:  # Short position
                # Calculate new trailing stop
                trailing_stop = self.lowest_price * (1 + self.trailing_stop_pct)
                
                # Only move stop loss down, never up
                if trailing_stop < self.stop_loss_price:
                    new_stop_loss = trailing_stop
                    logger.info(f"Trailing stop updated: {self.stop_loss_price:.2f} -> {new_stop_loss:.2f}")
                    self.stop_loss_price = new_stop_loss
        
        return new_stop_loss, should_exit
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate buy/sell signals based on LSTM predictions.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with signals
        """
        logger.info("Generating signals using LSTM strategy...")
        
        # Make a copy of the data
        df = data.copy()
        
        # Add returns column if not present
        if 'returns' not in df.columns:
            df['returns'] = df['close'].pct_change()
        
        # Train or retrain the model if needed
        if self.last_trained is None:
            self.train_model(df)
        
        # Get predictions
        predictions = self.predictor.predict(df)
        
        if predictions is None or predictions.empty:
            logger.error("No predictions generated")
            return df
        
        # Join predictions with original data
        df = df.join(predictions[['predicted_return', 'predicted_price']], how='left')
        
        # Calculate threshold
        if self.use_dynamic_threshold:
            threshold = self.calculate_dynamic_threshold(df)
        else:
            threshold = self.threshold
        
        # Initialize signals to 0
        df['signal'] = 0
        df['stop_loss'] = None
        df['take_profit'] = None
        
        # Initialize state variables
        in_position = 0  # 0: no position, 1: long, -1: short
        
        # Loop through data to generate signals with state management
        for i in range(len(df)):
            current_row = df.iloc[i]
            current_price = current_row['close']
            
            # Skip if no prediction
            if pd.isna(current_row['predicted_return']):
                df.loc[df.index[i], 'signal'] = in_position
                continue
            
            # Update stops if in position
            if in_position != 0:
                new_stop, should_exit = self.update_stops(current_price, in_position)
                
                if new_stop is not None:
                    df.loc[df.index[i], 'stop_loss'] = new_stop
                
                if should_exit:
                    # Exit position
                    df.loc[df.index[i], 'signal'] = 0
                    in_position = 0
                    
                    # Reset state
                    self.stop_loss_price = None
                    self.take_profit_price = None
                    self.open_position_price = None
                    self.highest_price = None
                    self.lowest_price = None
                    
                    continue
            
            # Generate signals based on predicted returns
            if in_position == 0:  # No position
                if current_row['predicted_return'] > threshold:
                    # Buy signal
                    df.loc[df.index[i], 'signal'] = 1
                    in_position = 1
                    
                    # Initialize position and risk management
                    self.open_position_price = current_price
                    self.highest_price = current_price
                    self.lowest_price = current_price
                    
                    # Calculate position size and stops
                    position_size = self.calculate_position_size(
                        data=df.iloc[:i+1],
                        entry_price=current_price
                    )
                    
                    # Store stop levels
                    df.loc[df.index[i], 'stop_loss'] = self.stop_loss_price
                    df.loc[df.index[i], 'take_profit'] = self.take_profit_price
                    
                elif current_row['predicted_return'] < -threshold:
                    # Sell signal
                    df.loc[df.index[i], 'signal'] = -1
                    in_position = -1
                    
                    # Initialize position and risk management
                    self.open_position_price = current_price
                    self.highest_price = current_price
                    self.lowest_price = current_price
                    
                    # Calculate position size and stops
                    position_size = self.calculate_position_size(
                        data=df.iloc[:i+1],
                        entry_price=current_price
                    )
                    
                    # For short positions, reverse stop and target
                    self.stop_loss_price = current_price * (1 + self.stop_loss_pct)
                    self.take_profit_price = current_price * (1 - self.take_profit_pct)
                    
                    # Store stop levels
                    df.loc[df.index[i], 'stop_loss'] = self.stop_loss_price
                    df.loc[df.index[i], 'take_profit'] = self.take_profit_price
                    
                else:
                    # No signal
                    df.loc[df.index[i], 'signal'] = 0
            
            else:
                # Maintain current position
                df.loc[df.index[i], 'signal'] = in_position
                
                # Store current stop loss
                df.loc[df.index[i], 'stop_loss'] = self.stop_loss_price
                df.loc[df.index[i], 'take_profit'] = self.take_profit_price
        
        # Generate position changes
        df['position'] = df['signal'].diff()
        
        # Fill NaN values
        df['position'] = df['position'].fillna(0)
        
        # Count signals
        buy_signals = (df['position'] == 1).sum() + (df['position'] == 2).sum()
        sell_signals = (df['position'] == -1).sum() + (df['position'] == -2).sum()
        
        logger.info(f"Generated {buy_signals} buy signals and {sell_signals} sell signals")
        
        return df
    
    def plot_signals(self, 
                  data: pd.DataFrame,
                  figsize: Tuple[int, int] = (12, 8),
                  save_path: Optional[str] = None) -> None:
        """
        Plot trading signals with predicted returns.
        
        Args:
            data: DataFrame with signals
            figsize: Figure size
            save_path: Path to save the plot
        """
        # Check if required columns exist
        required_columns = ['close', 'signal', 'predicted_return', 'stop_loss', 'take_profit']
        if not all(col in data.columns for col in required_columns):
            logger.error("Required columns missing for plotting")
            return
        
        plt.figure(figsize=figsize)
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True, 
                                     gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot price and signals
        ax1.plot(data.index, data['close'], label='Close Price', color='blue')
        
        # Plot buy signals
        buy_signals = data[data['position'] == 1].index
        ax1.scatter(buy_signals, data.loc[buy_signals, 'close'], marker='^', color='green', s=100, label='Buy')
        
        # Plot sell signals
        sell_signals = data[data['position'] == -1].index
        ax1.scatter(sell_signals, data.loc[sell_signals, 'close'], marker='v', color='red', s=100, label='Sell')
        
        # Plot exit signals
        exit_signals = data[(data['position'] == -2) | (data['position'] == 2)].index
        ax1.scatter(exit_signals, data.loc[exit_signals, 'close'], marker='o', color='purple', s=50, label='Exit')
        
        # Plot stop loss and take profit levels
        for col, color, label in [('stop_loss', 'red', 'Stop Loss'), ('take_profit', 'green', 'Take Profit')]:
            if col in data.columns:
                non_na = data[~data[col].isna()]
                if not non_na.empty:
                    ax1.scatter(non_na.index, non_na[col], marker='_', color=color, s=50, label=label)
        
        # Add labels and legend
        ax1.set_title('LSTM Trading Strategy', fontsize=16)
        ax1.set_ylabel('Price', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot predicted returns
        ax2.plot(data.index, data['predicted_return'], label='Predicted Return', color='blue')
        ax2.axhline(y=self.threshold, color='green', linestyle='--', label=f'Threshold ({self.threshold:.3f})')
        ax2.axhline(y=-self.threshold, color='red', linestyle='--')
        
        # Add labels and legend
        ax2.set_xlabel('Date', fontsize=14)
        ax2.set_ylabel('Predicted Return', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        
        # Save or show the plot
        if save_path:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            logger.info(f"Signal plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()