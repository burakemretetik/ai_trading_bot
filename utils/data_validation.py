# utils/data_validation.py - Data validation utilities
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any, Optional, Union, Callable
import logging
from datetime import datetime, timedelta

from utils.error_handling import ValidationError

logger = logging.getLogger(__name__)

def validate_ohlcv_dataframe(df: pd.DataFrame) -> Tuple[bool, str]:
    """
    Validate OHLCV data format.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check if DataFrame is empty
    if df.empty:
        return False, "DataFrame is empty"
    
    # Check required columns
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        return False, f"Missing required columns: {', '.join(missing_columns)}"
    
    # Check index type
    if not isinstance(df.index, pd.DatetimeIndex):
        return False, "Index is not DatetimeIndex"
    
    # Check for duplicate indices
    if df.index.duplicated().any():
        return False, "DataFrame contains duplicate indices"
    
    # Check for NaN values
    for col in required_columns:
        if df[col].isna().any():
            count = df[col].isna().sum()
            return False, f"Column '{col}' contains {count} NaN values"
    
    # Check for negative prices or volumes
    if (df['open'] <= 0).any():
        return False, "Column 'open' contains zero or negative values"
    if (df['high'] <= 0).any():
        return False, "Column 'high' contains zero or negative values"
    if (df['low'] <= 0).any():
        return False, "Column 'low' contains zero or negative values"
    if (df['close'] <= 0).any():
        return False, "Column 'close' contains zero or negative values"
    if (df['volume'] < 0).any():
        return False, "Column 'volume' contains negative values"
    
    # Check price relationships
    if not (df['high'] >= df['open']).all():
        return False, "Some 'high' values are less than 'open' values"
    if not (df['high'] >= df['close']).all():
        return False, "Some 'high' values are less than 'close' values"
    if not (df['low'] <= df['open']).all():
        return False, "Some 'low' values are greater than 'open' values"
    if not (df['low'] <= df['close']).all():
        return False, "Some 'low' values are greater than 'close' values"
    
    return True, ""


def validate_dataframe_features(df: pd.DataFrame, required_features: List[str]) -> Tuple[bool, str]:
    """
    Validate that DataFrame contains required features.
    
    Args:
        df: DataFrame to validate
        required_features: List of required feature columns
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check if DataFrame is empty
    if df.empty:
        return False, "DataFrame is empty"
    
    # Check required columns
    missing_features = [feat for feat in required_features if feat not in df.columns]
    if missing_features:
        return False, f"Missing required features: {', '.join(missing_features)}"
    
    # Check for NaN values in required features
    for feat in required_features:
        if df[feat].isna().any():
            count = df[feat].isna().sum()
            return False, f"Feature '{feat}' contains {count} NaN values"
    
    return True, ""


def check_for_data_anomalies(df: pd.DataFrame, 
                           column: str,
                           window: int = 20,
                           threshold: float = 3.0) -> pd.Series:
    """
    Detect anomalies in a DataFrame column using Z-score.
    
    Args:
        df: DataFrame to check
        column: Column name to check
        window: Rolling window size for Z-score calculation
        threshold: Z-score threshold for anomaly detection
        
    Returns:
        Boolean Series indicating anomalies
    """
    # Calculate rolling mean and standard deviation
    rolling_mean = df[column].rolling(window=window, min_periods=1).mean()
    rolling_std = df[column].rolling(window=window, min_periods=1).std()
    
    # Handle zero standard deviation
    rolling_std = rolling_std.replace(0, np.nan)
    rolling_std = rolling_std.fillna(df[column].std())
    
    # Calculate Z-score
    z_score = (df[column] - rolling_mean) / rolling_std
    
    # Detect anomalies
    anomalies = z_score.abs() > threshold
    
    if anomalies.any():
        anomaly_count = anomalies.sum()
        logger.warning(f"Detected {anomaly_count} anomalies in '{column}' using Z-score threshold {threshold}")
    
    return anomalies


def detect_price_gaps(df: pd.DataFrame, threshold: float = 0.05) -> pd.Series:
    """
    Detect significant price gaps between candles.
    
    Args:
        df: OHLCV DataFrame
        threshold: Threshold for gap detection (as percentage)
        
    Returns:
        Boolean Series indicating gaps
    """
    # Calculate previous close
    prev_close = df['close'].shift(1)
    
    # Calculate gap percentage
    gap_pct = (df['open'] - prev_close) / prev_close
    
    # Detect gaps
    gaps = gap_pct.abs() > threshold
    
    if gaps.any():
        gap_count = gaps.sum()
        logger.warning(f"Detected {gap_count} price gaps larger than {threshold:.1%}")
    
    return gaps


def detect_outliers_iqr(df: pd.DataFrame, column: str, multiplier: float = 1.5) -> pd.Series:
    """
    Detect outliers using Interquartile Range (IQR) method.
    
    Args:
        df: DataFrame to check
        column: Column name to check
        multiplier: IQR multiplier for outlier detection
        
    Returns:
        Boolean Series indicating outliers
    """
    # Calculate Q1, Q3, and IQR
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    
    # Define outlier bounds
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    
    # Detect outliers
    outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
    
    if outliers.any():
        outlier_count = outliers.sum()
        logger.warning(f"Detected {outlier_count} outliers in '{column}' using IQR method")
    
    return outliers


def validate_model_inputs(features: List[str],
                        target: str,
                        test_size: float) -> Tuple[bool, str]:
    """
    Validate model training inputs.
    
    Args:
        features: List of feature columns
        target: Target column name
        test_size: Test set size (between 0 and 1)
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check features
    if not features:
        return False, "Feature list is empty"
    
    # Check target
    if not target:
        return False, "Target is empty"
    
    # Check test_size
    if not 0 < test_size < 1:
        return False, f"Invalid test_size: {test_size}. Must be between 0 and 1"
    
    return True, ""


def validate_date_range(start_date: Optional[str],
                      end_date: Optional[str],
                      date_format: str = '%Y-%m-%d') -> Tuple[bool, str]:
    """
    Validate date range.
    
    Args:
        start_date: Start date string
        end_date: End date string
        date_format: Date format string
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    # If both dates are None, that's valid (use defaults)
    if start_date is None and end_date is None:
        return True, ""
    
    # Parse dates
    if start_date:
        try:
            start = datetime.strptime(start_date, date_format)
        except ValueError:
            return False, f"Invalid start_date format: {start_date}. Expected format: {date_format}"
    else:
        # Default to 1 year ago
        start = datetime.now() - timedelta(days=365)
    
    if end_date:
        try:
            end = datetime.strptime(end_date, date_format)
        except ValueError:
            return False, f"Invalid end_date format: {end_date}. Expected format: {date_format}"
    else:
        # Default to today
        end = datetime.now()
    
    # Check that start date is before end date
    if start >= end:
        return False, f"Start date ({start_date}) must be before end date ({end_date})"
    
    # Check that date range is not too large
    max_days = 3650  # 10 years
    days_diff = (end - start).days
    if days_diff > max_days:
        return False, f"Date range too large: {days_diff} days. Maximum allowed: {max_days} days"
    
    return True, ""


def validate_trading_parameters(
    initial_capital: float,
    commission: float,
    risk_per_trade: float,
    symbol: str
) -> Tuple[bool, str]:
    """
    Validate trading parameters.
    
    Args:
        initial_capital: Initial capital amount
        commission: Commission rate
        risk_per_trade: Risk per trade (as decimal)
        symbol: Trading symbol
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check initial capital
    if initial_capital <= 0:
        return False, f"Invalid initial_capital: {initial_capital}. Must be positive"
    
    # Check commission
    if commission < 0 or commission >= 1:
        return False, f"Invalid commission: {commission}. Must be between 0 and 1"
    
    # Check risk per trade
    if risk_per_trade <= 0 or risk_per_trade >= 1:
        return False, f"Invalid risk_per_trade: {risk_per_trade}. Must be between 0 and 1"
    
    # Check symbol
    if not symbol or not isinstance(symbol, str):
        return False, f"Invalid symbol: {symbol}"
    
    return True, ""