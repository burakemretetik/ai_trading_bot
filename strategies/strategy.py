# strategies/strategy.py - Base strategy class
import pandas as pd
import numpy as np
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class Strategy(ABC):
    """Base class for all trading strategies."""
    
    def __init__(self, name):
        """Initialize strategy with a name."""
        self.name = name
    
    @abstractmethod
    def generate_signals(self, data):
        """
        Generate trading signals from the data.
        
        Args:
            data: DataFrame with price data
            
        Returns:
            DataFrame with added signal columns
        """
        pass
    
    def calculate_position_size(self, capital, price, risk_per_trade=0.02, stop_loss_pct=0.05):
        """
        Calculate position size based on risk parameters.
        
        Args:
            capital: Available capital
            price: Current price
            risk_per_trade: Percentage of capital to risk per trade
            stop_loss_pct: Stop loss percentage
            
        Returns:
            Position size
        """
        # Calculate risk amount in currency
        risk_amount = capital * risk_per_trade
        
        # Calculate stop loss price
        stop_loss_price = price * (1 - stop_loss_pct)
        
        # Calculate risk per unit
        price_risk = price - stop_loss_price
        
        # Calculate position size
        position_size = risk_amount / price_risk
        
        return position_size