# utils/error_handling.py - Error handling utilities
import logging
import functools
import traceback
import time
from typing import Callable, Any, Optional, Type, Union, List, Dict, TypeVar

logger = logging.getLogger(__name__)

# Type variable for return type
T = TypeVar('T')

class TradingBotError(Exception):
    """Base exception class for all trading bot errors."""
    pass

class DataError(TradingBotError):
    """Error related to data fetching or processing."""
    pass

class APIError(TradingBotError):
    """Error related to API communication."""
    pass

class ModelError(TradingBotError):
    """Error related to ML model operations."""
    pass

class StrategyError(TradingBotError):
    """Error related to trading strategy execution."""
    pass

class ValidationError(TradingBotError):
    """Error related to data or parameter validation."""
    pass

def retry(max_tries: int = 3, 
        delay: float = 1.0, 
        backoff: float = 2.0, 
        exceptions: Union[Type[Exception], List[Type[Exception]]] = Exception) -> Callable:
    """
    Retry decorator with exponential backoff for functions.
    
    Args:
        max_tries: Maximum number of attempts
        delay: Initial delay between retries in seconds
        backoff: Backoff multiplier (e.g. 2 means delay doubles each retry)
        exceptions: Exception(s) to catch and retry on
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            mtries, mdelay = max_tries, delay
            
            while mtries > 1:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    logger.warning(f"Retrying {func.__name__} in {mdelay:.1f} seconds due to {e.__class__.__name__}: {str(e)}")
                    time.sleep(mdelay)
                    mtries -= 1
                    mdelay *= backoff
            
            # Last attempt
            return func(*args, **kwargs)
        return wrapper
    return decorator


def validate_dataframe(df_validator: Callable) -> Callable:
    """
    Decorator for validating pandas DataFrames.
    
    Args:
        df_validator: Function that takes a DataFrame and returns (is_valid, error_message)
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            result = func(*args, **kwargs)
            
            # Check if result is a DataFrame
            import pandas as pd
            if isinstance(result, pd.DataFrame):
                is_valid, error_message = df_validator(result)
                if not is_valid:
                    logger.error(f"DataFrame validation failed: {error_message}")
                    raise ValidationError(error_message)
            
            return result
        return wrapper
    return decorator


def handle_api_errors(func: Callable) -> Callable:
    """
    Decorator for handling API-related errors.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Check for common API errors
            error_str = str(e).lower()
            
            if 'timeout' in error_str or 'timed out' in error_str:
                logger.error(f"API timeout error: {e}")
                raise APIError(f"API request timed out: {e}")
            elif 'rate limit' in error_str:
                logger.error(f"API rate limit exceeded: {e}")
                raise APIError(f"Rate limit exceeded: {e}")
            elif 'authentication' in error_str or 'auth' in error_str:
                logger.error(f"API authentication error: {e}")
                raise APIError(f"Authentication error: {e}")
            elif 'permission' in error_str or 'forbidden' in error_str:
                logger.error(f"API permission error: {e}")
                raise APIError(f"Permission denied: {e}")
            else:
                logger.error(f"API error: {e}")
                raise APIError(f"API error: {e}")
    return wrapper


def safe_execute(default_return: Optional[T] = None, 
               log_exception: bool = True,
               reraise: bool = False) -> Callable:
    """
    Decorator for safely executing functions with default return on exception.
    
    Args:
        default_return: Value to return if an exception occurs
        log_exception: Whether to log the exception details
        reraise: Whether to re-raise the exception after logging
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_exception:
                    logger.error(f"Error in {func.__name__}: {e}")
                    logger.debug(f"Exception details: {traceback.format_exc()}")
                
                if reraise:
                    raise
                
                return default_return
        return wrapper
    return decorator


def log_execution_time(func: Callable) -> Callable:
    """
    Decorator for logging function execution time.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        execution_time = end_time - start_time
        logger.debug(f"{func.__name__} executed in {execution_time:.4f} seconds")
        
        return result
    return wrapper


def validate_parameters(**validators: Callable) -> Callable:
    """
    Decorator for validating function parameters.
    
    Args:
        **validators: Dictionary of parameter validators (param_name: validator_func)
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Get function signature
            import inspect
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # Validate parameters
            for param_name, validator in validators.items():
                if param_name in bound_args.arguments:
                    value = bound_args.arguments[param_name]
                    is_valid = validator(value)
                    if not is_valid:
                        raise ValidationError(f"Invalid parameter '{param_name}': {value}")
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


# Example parameter validators
def is_positive(value: Union[int, float]) -> bool:
    """Check if a value is positive."""
    return value > 0

def is_in_range(min_val: Union[int, float], max_val: Union[int, float]) -> Callable:
    """Create a validator for checking if a value is in range."""
    def validator(value: Union[int, float]) -> bool:
        return min_val <= value <= max_val
    return validator

def is_one_of(valid_values: List[Any]) -> Callable:
    """Create a validator for checking if a value is one of a list of valid values."""
    def validator(value: Any) -> bool:
        return value in valid_values
    return validator