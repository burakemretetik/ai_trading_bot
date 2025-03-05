# utils/config_manager.py - Configuration management
import os
import yaml
import json
import logging
from typing import Dict, Any, Optional, Union, List, Tuple
from dotenv import load_dotenv
from dataclasses import dataclass, field
import argparse

from utils.error_handling import ValidationError
from utils.constants import TradingMode, StrategyType, DataSource, Timeframe

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for ML models."""
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    test_size: float = 0.2
    threshold: float = 0.002
    retrain_period: int = 30  # days


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    initial_capital: float = 10000.0
    commission: float = 0.001  # 0.1%
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    enable_plotting: bool = True
    save_results: bool = False
    output_dir: str = "results"


@dataclass
class TradingConfig:
    """Configuration for live trading."""
    symbol: str = "BTCUSDT"
    timeframe: str = "1h"
    risk_per_trade: float = 0.02  # 2% risk per trade
    max_open_trades: int = 3
    stop_loss_pct: float = 0.05  # 5% stop loss
    take_profit_pct: float = 0.1  # 10% take profit
    trailing_stop: bool = False
    trailing_stop_distance: float = 0.02  # 2% trailing stop


@dataclass
class ApiConfig:
    """Configuration for API credentials."""
    api_key: str = ""
    api_secret: str = ""
    testnet: bool = True
    exchange: str = "binance"


@dataclass
class LoggingConfig:
    """Configuration for logging."""
    log_level: str = "INFO"
    log_dir: str = "logs"
    console_logging: bool = True
    file_logging: bool = True
    json_logging: bool = False
    max_file_size: int = 10 * 1024 * 1024  # 10 MB
    backup_count: int = 5


@dataclass
class Config:
    """Main configuration class."""
    # General settings
    mode: TradingMode = TradingMode.BACKTEST
    strategy: StrategyType = StrategyType.TORCH_ML
    data_source: DataSource = DataSource.CCXT
    
    # Component configurations
    model: ModelConfig = field(default_factory=ModelConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    api: ApiConfig = field(default_factory=ApiConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    # Additional settings
    debug: bool = False
    verbose: bool = False
    config_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "mode": self.mode.value,
            "strategy": self.strategy.value,
            "data_source": self.data_source.value,
            "model": {
                "hidden_dim": self.model.hidden_dim,
                "num_layers": self.model.num_layers,
                "dropout": self.model.dropout,
                "learning_rate": self.model.learning_rate,
                "batch_size": self.model.batch_size,
                "epochs": self.model.epochs,
                "test_size": self.model.test_size,
                "threshold": self.model.threshold,
                "retrain_period": self.model.retrain_period
            },
            "backtest": {
                "initial_capital": self.backtest.initial_capital,
                "commission": self.backtest.commission,
                "start_date": self.backtest.start_date,
                "end_date": self.backtest.end_date,
                "enable_plotting": self.backtest.enable_plotting,
                "save_results": self.backtest.save_results,
                "output_dir": self.backtest.output_dir
            },
            "trading": {
                "symbol": self.trading.symbol,
                "timeframe": self.trading.timeframe,
                "risk_per_trade": self.trading.risk_per_trade,
                "max_open_trades": self.trading.max_open_trades,
                "stop_loss_pct": self.trading.stop_loss_pct,
                "take_profit_pct": self.trading.take_profit_pct,
                "trailing_stop": self.trading.trailing_stop,
                "trailing_stop_distance": self.trading.trailing_stop_distance
            },
            "api": {
                "exchange": self.api.exchange,
                "testnet": self.api.testnet,
                # Don't include API credentials in output
            },
            "logging": {
                "log_level": self.logging.log_level,
                "log_dir": self.logging.log_dir,
                "console_logging": self.logging.console_logging,
                "file_logging": self.logging.file_logging,
                "json_logging": self.logging.json_logging,
                "max_file_size": self.logging.max_file_size,
                "backup_count": self.logging.backup_count
            },
            "debug": self.debug,
            "verbose": self.verbose,
            "config_path": self.config_path
        }

    def save_to_file(self, filepath: str) -> bool:
        """Save configuration to file."""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Save config to file
            config_dict = self.to_dict()
            
            # Don't save API credentials
            if "api" in config_dict:
                if "api_key" in config_dict["api"]:
                    config_dict["api"]["api_key"] = ""
                if "api_secret" in config_dict["api"]:
                    config_dict["api"]["api_secret"] = ""
            
            # Determine file format from extension
            ext = os.path.splitext(filepath)[1].lower()
            if ext == ".json":
                with open(filepath, "w") as f:
                    json.dump(config_dict, f, indent=2)
            else:  # Default to YAML
                with open(filepath, "w") as f:
                    yaml.dump(config_dict, f, default_flow_style=False)
            
            logger.info(f"Configuration saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            return False


class ConfigManager:
    """Manager for loading, validating, and accessing configuration."""
    
    def __init__(self):
        """Initialize ConfigManager."""
        self.config = Config()
        self.loaded = False
    
    def load_from_env(self) -> None:
        """Load configuration from environment variables."""
        # Load .env file if present
        load_dotenv()
        
        # Load API credentials
        self.config.api.api_key = os.getenv("API_KEY", "")
        self.config.api.api_secret = os.getenv("API_SECRET", "")
        self.config.api.testnet = os.getenv("TESTNET", "True").lower() in ["true", "1", "yes"]
        self.config.api.exchange = os.getenv("EXCHANGE", "binance")
        
        # Load other settings with env prefix TB_
        if os.getenv("TB_MODE"):
            self.config.mode = TradingMode(os.getenv("TB_MODE"))
        
        if os.getenv("TB_STRATEGY"):
            self.config.strategy = StrategyType(os.getenv("TB_STRATEGY"))
        
        if os.getenv("TB_DATA_SOURCE"):
            self.config.data_source = DataSource(os.getenv("TB_DATA_SOURCE"))
        
        if os.getenv("TB_SYMBOL"):
            self.config.trading.symbol = os.getenv("TB_SYMBOL")
        
        if os.getenv("TB_TIMEFRAME"):
            self.config.trading.timeframe = os.getenv("TB_TIMEFRAME")
        
        # Log level
        if os.getenv("TB_LOG_LEVEL"):
            self.config.logging.log_level = os.getenv("TB_LOG_LEVEL")
        
        logger.debug("Configuration loaded from environment variables")
    
    def load_from_file(self, filepath: str) -> bool:
        """
        Load configuration from file.
        
        Args:
            filepath: Path to configuration file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if file exists
            if not os.path.exists(filepath):
                logger.error(f"Configuration file not found: {filepath}")
                return False
            
            # Determine file format from extension
            ext = os.path.splitext(filepath)[1].lower()
            if ext == ".json":
                with open(filepath, "r") as f:
                    config_dict = json.load(f)
            else:  # Default to YAML
                with open(filepath, "r") as f:
                    config_dict = yaml.safe_load(f)
            
            # Update configuration
            self._update_from_dict(config_dict)
            
            # Store config path
            self.config.config_path = filepath
            
            logger.info(f"Configuration loaded from {filepath}")
            self.loaded = True
            return True
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            return False
    
    def load_from_args(self, args: argparse.Namespace) -> None:
        """
        Load configuration from command-line arguments.
        
        Args:
            args: Parsed command-line arguments
        """
        # Update configuration from args
        for key, value in vars(args).items():
            if value is None:
                continue
            
            # Handle config path separately
            if key == "config":
                if value:
                    self.load_from_file(value)
                continue
            
            # Convert snake_case to attribute paths (e.g., model_hidden_dim -> model.hidden_dim)
            if "_" in key:
                parts = key.split("_")
                
                # Try to find the first level component
                component = None
                for i in range(1, len(parts)):
                    section = "_".join(parts[:i])
                    if hasattr(self.config, section):
                        component = getattr(self.config, section)
                        attr_name = "_".join(parts[i:])
                        break
                
                # If found component, update attribute
                if component and hasattr(component, attr_name):
                    setattr(component, attr_name, value)
                else:
                    # Try direct attribute
                    if hasattr(self.config, key):
                        setattr(self.config, key, value)
            else:
                # Direct attribute
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
        
        logger.debug("Configuration updated from command-line arguments")
    
    def _update_from_dict(self, config_dict: Dict[str, Any]) -> None:
        """
        Update configuration from dictionary.
        
        Args:
            config_dict: Configuration dictionary
        """
        # Update main settings
        if "mode" in config_dict:
            self.config.mode = TradingMode(config_dict["mode"])
        
        if "strategy" in config_dict:
            self.config.strategy = StrategyType(config_dict["strategy"])
        
        if "data_source" in config_dict:
            self.config.data_source = DataSource(config_dict["data_source"])
        
        # Update component configurations
        for component in ["model", "backtest", "trading", "api", "logging"]:
            if component in config_dict:
                component_config = getattr(self.config, component)
                for key, value in config_dict[component].items():
                    if hasattr(component_config, key):
                        setattr(component_config, key, value)
        
        # Update other settings
        for key in ["debug", "verbose"]:
            if key in config_dict:
                setattr(self.config, key, config_dict[key])
    
    def validate_config(self) -> Tuple[bool, List[str]]:
        """
        Validate configuration.
        
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Validate mode
        if not isinstance(self.config.mode, TradingMode):
            errors.append(f"Invalid mode: {self.config.mode}")
        
        # Validate strategy
        if not isinstance(self.config.strategy, StrategyType):
            errors.append(f"Invalid strategy: {self.config.strategy}")
        
        # Validate data source
        if not isinstance(self.config.data_source, DataSource):
            errors.append(f"Invalid data source: {self.config.data_source}")
        
        # Validate model config
        if self.config.model.hidden_dim <= 0:
            errors.append(f"Invalid hidden_dim: {self.config.model.hidden_dim}")
        if self.config.model.num_layers <= 0:
            errors.append(f"Invalid num_layers: {self.config.model.num_layers}")
        if not 0 <= self.config.model.dropout < 1:
            errors.append(f"Invalid dropout: {self.config.model.dropout}")
        if self.config.model.learning_rate <= 0:
            errors.append(f"Invalid learning_rate: {self.config.model.learning_rate}")
        if self.config.model.batch_size <= 0:
            errors.append(f"Invalid batch_size: {self.config.model.batch_size}")
        if self.config.model.epochs <= 0:
            errors.append(f"Invalid epochs: {self.config.model.epochs}")
        if not 0 < self.config.model.test_size < 1:
            errors.append(f"Invalid test_size: {self.config.model.test_size}")
        
        # Validate backtest config
        if self.config.backtest.initial_capital <= 0:
            errors.append(f"Invalid initial_capital: {self.config.backtest.initial_capital}")
        if not 0 <= self.config.backtest.commission < 1:
            errors.append(f"Invalid commission: {self.config.backtest.commission}")
        
        # Validate trading config
        if not self.config.trading.symbol:
            errors.append("Symbol is required")
        if not self.config.trading.timeframe:
            errors.append("Timeframe is required")
        if not 0 < self.config.trading.risk_per_trade < 1:
            errors.append(f"Invalid risk_per_trade: {self.config.trading.risk_per_trade}")
        
        # Validate API config if in live mode
        if self.config.mode == TradingMode.LIVE:
            if not self.config.api.api_key:
                errors.append("API key is required for live trading")
            if not self.config.api.api_secret:
                errors.append("API secret is required for live trading")
        
        return len(errors) == 0, errors
    
    def get_config(self) -> Config:
        """
        Get the current configuration.
        
        Returns:
            Current Config object
        """
        return self.config
    
    def print_config(self, include_secrets: bool = False) -> None:
        """
        Print current configuration.
        
        Args:
            include_secrets: Whether to include API credentials
        """
        config_dict = self.config.to_dict()
        
        # Redact API credentials unless explicitly requested
        if not include_secrets and "api" in config_dict:
            if "api_key" in config_dict["api"] and config_dict["api"]["api_key"]:
                config_dict["api"]["api_key"] = "****"
            if "api_secret" in config_dict["api"] and config_dict["api"]["api_secret"]:
                config_dict["api"]["api_secret"] = "****"
        
        print(json.dumps(config_dict, indent=2))
    
    def create_default_config_file(self, filepath: str) -> bool:
        """
        Create a default configuration file.
        
        Args:
            filepath: Path to create configuration file
            
        Returns:
            True if successful, False otherwise
        """
        # Create default configuration
        default_config = Config()
        
        # Save to file
        return default_config.save_to_file(filepath)


# Singleton instance
_config_manager = None

def get_config_manager() -> ConfigManager:
    """
    Get or create the singleton ConfigManager instance.
    
    Returns:
        ConfigManager instance
    """
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
        # Load from environment variables by default
        _config_manager.load_from_env()
    
    return _config_manager