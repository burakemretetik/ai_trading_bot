# AI Trading Bot

A PyTorch-based trading bot for cryptocurrency markets with advanced backtesting capabilities, machine learning-driven strategies, and comprehensive risk management.

![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## Features

- **Advanced Strategy Implementation**:
  - Neural network-based price prediction using PyTorch
  - Simple Moving Average (SMA) strategy with noise filtering
  - Optimizable strategy parameters

- **Comprehensive Backtesting**:
  - Performance metrics (Sharpe, Sortino, drawdown, win rate, etc.)
  - Monte Carlo simulations for robustness testing
  - Slippage and commission modeling
  - Strategy comparison tools

- **Risk Management**:
  - Position sizing based on account risk
  - Stop-loss implementation
  - Maximum position limits

- **Data Sources**:
  - Real market data via CCXT (supports many exchanges)
  - Yahoo Finance integration
  - CSV data import
  - Simulated data generation for testing

- **Visualization**:
  - Performance charts and trade analysis
  - Equity curves and drawdown visualization
  - Return distribution analysis

## Project Structure

```
ai_trading_bot/
├── config/               # Configuration files
│   └── default_config.yaml   # Default configuration
├── data/                 # Data storage
├── models/               # ML models
│   ├── predictor.py      # PyTorch price predictor
├── strategies/           # Trading strategies
│   ├── strategy.py       # Base strategy class
│   ├── sma_strategy.py   # SMA implementation
│   ├── ml_strategy.py    # ML-based strategy
├── utils/                # Utility functions
│   ├── data.py           # Data acquisition and processing
│   ├── backtester.py     # Backtesting framework
│   ├── visualizer.py     # Visualization tools
├── tests/                # Unit and integration tests
├── main.py               # Entry point
├── config.py             # Configuration loader
├── requirements.txt      # Dependencies
└── README.md             # This file
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ai_trading_bot.git
cd ai_trading_bot
```

2. Create a virtual environment:
```bash
python -m venv venv
```

3. Activate the virtual environment:
   - On Windows:
   ```bash
   .\venv\Scripts\activate
   ```
   - On macOS/Linux:
   ```bash
   source venv/bin/activate
   ```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

5. Create a `.env` file with your API credentials:
```
API_KEY=your_exchange_api_key
API_SECRET=your_exchange_api_secret
```

## Configuration

The bot is configured using YAML files in the `config/` directory, with `default_config.yaml` as the base. You can override settings by:

1. Modifying the default config file
2. Creating a custom config file and specifying it with the `--config` parameter
3. Using command-line arguments to override specific settings

### Key Configuration Sections:

- **General**: Mode, strategy, and data source
- **Model**: Neural network parameters
- **SMA Strategy**: Moving average window sizes and filtering options
- **Risk Management**: Risk per trade, stop-loss, position sizing
- **Backtesting**: Capital, commission, data range
- **Trading**: Symbol, timeframe, refresh intervals
- **API**: Exchange settings
- **Logging**: Log levels and outputs

## Usage

### Backtesting with Different Strategies

```bash
# Simple Moving Average strategy backtest
python main.py --mode backtest --strategy sma --symbol BTCUSDT --timeframe 1h

# Machine Learning strategy backtest
python main.py --mode backtest --strategy ml --symbol BTCUSDT --timeframe 1h

# Using custom SMA parameters
python main.py --mode backtest --strategy sma --short-window 10 --long-window 30

# Backtest with custom date range
python main.py --mode backtest --start-date 2023-01-01 --end-date 2023-06-30
```

### Strategy Comparison

Compare multiple strategies on the same dataset:

```bash
python main.py --mode compare --symbol BTCUSDT --timeframe 1d
```

This will run different strategies (SMA and ML variations) on the same data and produce comparative performance reports.

### Strategy Optimization

Find optimal parameters for a strategy:

```bash
python main.py --mode optimize --strategy sma --symbol BTCUSDT --timeframe 1d
```

This performs a grid search to find the best parameters (e.g., moving average windows for SMA).

### Simulated Data Testing

Test your strategies with generated data:

```bash
python main.py --mode simulated --strategy sma
```

### Paper Trading

Run the bot in paper trading mode (simulated trading with real market data):

```bash
python main.py --mode paper --strategy ml --symbol BTCUSDT --timeframe 1h
```

### Live Trading (use with caution)

```bash
python main.py --mode live --strategy ml --symbol BTCUSDT --timeframe 1h
```

**WARNING**: Live trading uses real funds. Extensive testing in paper mode is recommended before live deployment.

## Risk Management

The bot implements several risk management techniques:

1. **Position Sizing**: Limits the amount of capital at risk per trade
   ```yaml
   risk:
     risk_per_trade: 0.02  # Risk 2% of capital per trade
   ```

2. **Stop Loss**: Automatically exits positions at a defined loss threshold
   ```yaml
   risk:
     stop_loss: 0.05  # 5% stop loss
   ```

3. **Position Limits**: Caps the maximum position size
   ```yaml
   risk:
     max_position_size: 0.2  # Max 20% of capital in one position
   ```

4. **Slippage Modeling**: Accounts for execution slippage in backtest results
   ```yaml
   backtest:
     slippage: 0.0005  # 0.05% slippage
   ```

## Interpreting Backtest Results

The backtest results include several key metrics:

- **Total Return**: Overall percentage return
- **Annualized Return**: Return normalized to a yearly rate
- **Sharpe Ratio**: Risk-adjusted return (higher is better, >1 is good)
- **Sortino Ratio**: Downside risk-adjusted return
- **Max Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Gross profit divided by gross loss (>1 is profitable)

Example output:
```
Backtest Performance:
  total_return: 24.56%
  annualized_return: 31.78%
  sharpe_ratio: 1.4532
  sortino_ratio: 2.1236
  max_drawdown: -12.45%
  win_rate: 62.35%
  profit_factor: 1.87
  total_trades: 85
```

### Monte Carlo Analysis

The Monte Carlo analysis shows the robustness of your strategy by simulating many possible outcomes:

```
Monte Carlo Results:
  median_final_equity: $12,345.67
  95% CI for final equity: $10,123.45 to $14,567.89
  mean_max_drawdown: -15.67%
  worst_case_drawdown: -25.43%
  probability of profit: 87.65%
```

This helps assess whether your strategy's performance is likely due to skill or luck.

## ML Strategy Notes

The machine learning strategy uses a PyTorch neural network to predict price movements. Key considerations:

1. **Feature Engineering**: The model uses various technical indicators
2. **Prediction Threshold**: Only trades when the predicted return exceeds the threshold
3. **Retraining**: The model is periodically retrained to adapt to market changes
4. **Confidence Filtering**: Trades are only executed when prediction confidence is high

Configure these aspects in the `model` section of the config:

```yaml
model:
  threshold: 0.002
  retrain_period: 30
  feature_engineering: advanced
  min_confidence: 0.6
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This software is for educational purposes only. Use at your own risk. Cryptocurrency trading involves significant risk and can result in the loss of your invested capital. Past performance is not indicative of future results.