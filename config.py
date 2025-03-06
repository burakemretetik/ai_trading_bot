# config.py - Enhanced configuration
import os
import yaml
import logging
from typing import Dict, Any
from dotenv import load_dotenv  # Added python-dotenv dependency

# Setup logging
logger = logging.getLogger(__name__)

def deep_merge(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge dictionaries.
    
    Args:
        dict1: Base dictionary
        dict2: Dictionary to merge on top of base
        
    Returns:
        Merged dictionary
    """
    result = dict1.copy()
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result

def load_config(config_path="config/default_config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except FileNotFoundError:
        logger.warning(f"Config file {config_path} not found, using defaults")
        return {}
    except Exception as e:
        logger.error(f"Error loading config file: {e}")
        return {}

# Load environment variables using python-dotenv
dotenv_loaded = load_dotenv()
if not dotenv_loaded:
    logger.warning(".env file not found or could not be loaded")

# Default configuration
DEFAULT_CONFIG = {
    'mode': 'backtest',
    'strategy': 'sma',
    'data_source': 'ccxt',
    'symbol': 'BTCUSDT',
    'timeframe': '1h',
    'initial_capital': 10000.0,
    'commission': 0.001,
    'risk_per_trade': 0.02,
    'model': {
        'hidden_dim': 64,
        'num_layers': 2,
        'dropout': 0.2,
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 100,
        'test_size': 0.2,
        'threshold': 0.002,
        'retrain_period': 30
    },
    'backtest': {
        'start_date': None,
        'end_date': None,
        'enable_plotting': True,
        'save_results': True,
        'output_dir': 'results'
    },
    'logging': {
        'log_level': 'INFO',
        'log_dir': 'logs',
        'console_logging': True,
        'file_logging': True
    }
}

# Load user configuration
USER_CONFIG = load_config()

# Merge configurations using deep merge
CONFIG = deep_merge(DEFAULT_CONFIG, USER_CONFIG)

# Extract key settings for easy access
MODE = CONFIG['mode']
STRATEGY = CONFIG['strategy']
DATA_SOURCE = CONFIG['data_source']
SYMBOL = CONFIG['symbol']
TIMEFRAME = CONFIG['timeframe']
INITIAL_CAPITAL = CONFIG['initial_capital']
COMMISSION = CONFIG['commission']
RISK_PER_TRADE = CONFIG['risk_per_trade']

# API credentials (from environment variables)
API_KEY = os.getenv('API_KEY', '')
API_SECRET = os.getenv('API_SECRET', '')

# Setup logging
def setup_logging():
    """Set up logging configuration."""
    log_level = getattr(logging, CONFIG['logging']['log_level'])
    log_dir = CONFIG['logging']['log_dir']
    
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Create handlers
    handlers = []
    
    # Console handler
    if CONFIG['logging']['console_logging']:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        handlers.append(console_handler)
    
    # File handler
    if CONFIG['logging']['file_logging']:
        log_file = os.path.join(log_dir, f"trading_bot_{MODE}_{STRATEGY}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        handlers.append(file_handler)
    
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    
    logger.info(f"Logging initialized. Log level: {CONFIG['logging']['log_level']}")

# Initialize logging
setup_logging()