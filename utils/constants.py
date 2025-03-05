# utils/constants.py - Constants and type definitions
from enum import Enum
from typing import Dict, List, Tuple, Union, TypedDict, Optional

# Trading modes
class TradingMode(str, Enum):
    """Trading operation modes."""
    BACKTEST = "backtest"
    PAPER = "paper"
    LIVE = "live"
    SIMULATED = "simulated"
    RECENT = "recent"
    YFINANCE = "yfinance"

# Strategy types
class StrategyType(str, Enum):
    """Available trading strategies."""
    SMA = "sma"
    ML = "ml"
    TORCH_ML = "torch_ml"

# Data sources
class DataSource(str, Enum):
    """Available data sources."""
    CCXT = "ccxt"
    YFINANCE = "yfinance"
    SIMULATED = "simulated"
    CSV = "csv"

# Timeframes
class Timeframe(str, Enum):
    """Available timeframes."""
    MINUTE_1 = "1m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    MINUTE_30 = "30m"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    DAY_1 = "1d"
    WEEK_1 = "1w"
    MONTH_1 = "1mo"

# Signal types
class SignalType(int, Enum):
    """Trading signal types."""
    SELL = -1
    NEUTRAL = 0
    BUY = 1

# Position change types
class PositionChangeType(int, Enum):
    """Position change types."""
    SELL_TO_NEUTRAL = -1
    NO_CHANGE = 0
    BUY_TO_NEUTRAL = 1
    NEUTRAL_TO_SELL = -2
    NEUTRAL_TO_BUY = 2

# Performance metrics
class PerformanceMetrics(TypedDict, total=False):
    """Performance metrics dictionary type."""
    initial_equity: float
    final_equity: float
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_profit: float
    avg_loss: float
    avg_trade: float
    max_profit: float
    max_loss: float
    avg_hold_time: float
    start_date: str
    end_date: str

# Model training results
class ModelTrainingResults(TypedDict, total=False):
    """Model training results dictionary type."""
    train_loss: float
    test_loss: float
    r2_train: float
    r2_test: float
    feature_importance: object  # DataFrame
    training_date: str
    model_type: str
    hyperparameters: Dict[str, Union[int, float, str]]

# Trade dictionary
class Trade(TypedDict, total=False):
    """Trade information dictionary type."""
    id: str
    symbol: str
    type: str  # 'buy' or 'sell'
    entry_time: str
    entry_price: float
    exit_time: Optional[str]
    exit_price: Optional[float]
    quantity: float
    fees: float
    pnl: Optional[float]
    pnl_percent: Optional[float]
    status: str  # 'open' or 'closed'
    stop_loss: Optional[float]
    take_profit: Optional[float]

# Feature groups for technical analysis
PRICE_FEATURES = [
    'returns', 
    'log_returns'
]

MOVING_AVERAGE_FEATURES = [
    'ma_5', 'ma_10', 'ma_20', 'ma_50', 'ma_100',
    'ma_ratio_5', 'ma_ratio_10', 'ma_ratio_20', 'ma_ratio_50', 'ma_ratio_100',
    'ema_5', 'ema_10', 'ema_20', 'ema_50', 'ema_100',
    'ema_ratio_5', 'ema_ratio_10', 'ema_ratio_20', 'ema_ratio_50', 'ema_ratio_100'
]

VOLATILITY_FEATURES = [
    'volatility_5', 'volatility_10', 'volatility_20'
]

OSCILLATOR_FEATURES = [
    'rsi_14', 'macd', 'macd_signal', 'macd_hist'
]

VOLUME_FEATURES = [
    'volume_ma_5', 'volume_ratio', 'obv'
]

CHANNEL_FEATURES = [
    'upper_channel_10', 'lower_channel_10', 'channel_width_10', 'channel_position_10',
    'upper_channel_20', 'lower_channel_20', 'channel_width_20', 'channel_position_20'
]

# Exchange rate limits (requests per minute)
EXCHANGE_RATE_LIMITS: Dict[str, int] = {
    'binance': 1200,
    'coinbase': 300,
    'kraken': 60,
    'kucoin': 180,
    'bitfinex': 60,
    'bitstamp': 60,
    'gemini': 120,
    'huobi': 100,
    'okex': 120,
    'ftx': 60
}

# Standard data columns
OHLCV_COLUMNS = ['open', 'high', 'low', 'close', 'volume']