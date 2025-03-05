# utils/logging_config.py - Enhanced logging configuration
import logging
import logging.handlers
import os
import json
import time
from datetime import datetime
from typing import Dict, Any, Optional, Union, List
import traceback
import sys
import platform

# Define custom log levels
TRADE = 25  # Between INFO and WARNING
logging.addLevelName(TRADE, "TRADE")

# Define log formatters
SIMPLE_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DETAILED_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(pathname)s:%(lineno)d - %(message)s"
JSON_FORMAT = '{"time": "%(asctime)s", "name": "%(name)s", "level": "%(levelname)s", "message": "%(message)s"}'

class JsonFormatter(logging.Formatter):
    """Format log records as JSON."""
    
    def format(self, record):
        """Format log record as JSON."""
        # Create log record dict
        log_record = {
            "timestamp": self.formatTime(record),
            "name": record.name,
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_record["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info)
            }
        
        # Add extra attributes
        for key, value in record.__dict__.items():
            if key not in ["args", "asctime", "created", "exc_info", "exc_text", "filename",
                          "funcName", "id", "levelname", "levelno", "lineno", "module",
                          "msecs", "message", "msg", "name", "pathname", "process",
                          "processName", "relativeCreated", "stack_info", "thread", "threadName"]:
                log_record[key] = value
        
        return json.dumps(log_record)


class TradeLogger(logging.Logger):
    """Custom logger with trade method."""
    
    def trade(self, msg, *args, **kwargs):
        """Log a trade message with TRADE level."""
        if self.isEnabledFor(TRADE):
            self._log(TRADE, msg, args, **kwargs)


def setup_logging(
    log_level: Union[int, str] = logging.INFO,
    log_dir: str = "logs",
    app_name: str = "trading_bot",
    console: bool = True,
    file_logging: bool = True,
    json_logging: bool = False,
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 5
) -> logging.Logger:
    """
    Configure comprehensive logging system.
    
    Args:
        log_level: Logging level (default: INFO)
        log_dir: Directory for log files
        app_name: Application name for log files
        console: Whether to log to console
        file_logging: Whether to log to file
        json_logging: Whether to use JSON format for file logs
        max_bytes: Maximum file size before rotation
        backup_count: Number of backup files to keep
        
    Returns:
        Configured root logger
    """
    # Create logs directory if it doesn't exist
    if file_logging and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Get root logger
    root_logger = logging.getLogger()
    
    # Set level
    if isinstance(log_level, str):
        log_level = getattr(logging, log_level.upper())
    root_logger.setLevel(log_level)
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Setup handlers
    handlers = []
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(SIMPLE_FORMAT))
        handlers.append(console_handler)
    
    # File handlers
    if file_logging:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Regular log file
        log_file = os.path.join(log_dir, f"{app_name}_{timestamp}.log")
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        
        if json_logging:
            file_handler.setFormatter(JsonFormatter())
        else:
            file_handler.setFormatter(logging.Formatter(DETAILED_FORMAT))
        
        handlers.append(file_handler)
        
        # Error log file
        error_log_file = os.path.join(log_dir, f"{app_name}_{timestamp}_error.log")
        error_file_handler = logging.handlers.RotatingFileHandler(
            error_log_file,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        error_file_handler.setLevel(logging.ERROR)
        error_file_handler.setFormatter(logging.Formatter(DETAILED_FORMAT))
        handlers.append(error_file_handler)
        
        # Trade log file
        trade_log_file = os.path.join(log_dir, f"{app_name}_{timestamp}_trades.log")
        trade_file_handler = logging.handlers.RotatingFileHandler(
            trade_log_file,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        trade_file_handler.setLevel(TRADE)
        trade_file_handler.setFormatter(logging.Formatter(DETAILED_FORMAT))
        handlers.append(trade_file_handler)
    
    # Add all handlers to logger
    for handler in handlers:
        root_logger.addHandler(handler)
    
    # Register custom logger class
    logging.setLoggerClass(TradeLogger)
    
    # Log system info
    logger = logging.getLogger("system")
    logger.info(f"Logging initialized with level {logging.getLevelName(log_level)}")
    logger.info(f"Python version: {platform.python_version()}")
    logger.info(f"Platform: {platform.platform()}")
    
    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get logger with custom trade method.
    
    Args:
        name: Logger name
        
    Returns:
        Logger with trade method
    """
    logger = logging.getLogger(name)
    
    # Add trade method if it doesn't exist
    if not hasattr(logger, 'trade'):
        def trade(msg, *args, **kwargs):
            if logger.isEnabledFor(TRADE):
                logger._log(TRADE, msg, args, **kwargs)
        logger.trade = trade
    
    return logger


def log_dataframe_info(df: Any, name: str, logger: Optional[logging.Logger] = None) -> None:
    """
    Log information about a DataFrame.
    
    Args:
        df: DataFrame to log info about
        name: Name for the DataFrame
        logger: Logger to use (default: get new logger)
    """
    if logger is None:
        logger = get_logger("dataframe")
    
    try:
        import pandas as pd
        if isinstance(df, pd.DataFrame):
            info = {
                "name": name,
                "shape": df.shape,
                "columns": df.columns.tolist(),
                "index_type": str(type(df.index)),
                "memory_usage": df.memory_usage(deep=True).sum() / (1024 * 1024),  # MB
                "missing_values": df.isna().sum().to_dict(),
                "first_timestamp": str(df.index.min()) if isinstance(df.index, pd.DatetimeIndex) else "not datetime",
                "last_timestamp": str(df.index.max()) if isinstance(df.index, pd.DatetimeIndex) else "not datetime",
            }
            logger.debug(f"DataFrame '{name}' info: {json.dumps(info)}")
        else:
            logger.warning(f"Object '{name}' is not a pandas DataFrame")
    except Exception as e:
        logger.warning(f"Error logging DataFrame info: {e}")


def log_trade(
    symbol: str,
    action: str,
    price: float,
    quantity: float,
    timestamp: Optional[datetime] = None,
    trade_id: Optional[str] = None,
    reason: Optional[str] = None,
    logger: Optional[logging.Logger] = None
) -> None:
    """
    Log a trade with TRADE level.
    
    Args:
        symbol: Trading symbol
        action: Trade action ('BUY' or 'SELL')
        price: Trade price
        quantity: Trade quantity
        timestamp: Trade timestamp (default: now)
        trade_id: Trade ID (default: None)
        reason: Trade reason (default: None)
        logger: Logger to use (default: get new logger)
    """
    if logger is None:
        logger = get_logger("trades")
    
    if timestamp is None:
        timestamp = datetime.now()
    
    trade_info = {
        "symbol": symbol,
        "action": action.upper(),
        "price": price,
        "quantity": quantity,
        "value": price * quantity,
        "timestamp": timestamp.isoformat(),
        "trade_id": trade_id,
        "reason": reason
    }
    
    logger.trade(json.dumps(trade_info))