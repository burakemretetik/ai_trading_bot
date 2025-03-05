# AI Trading Bot

A PyTorch-based trading bot for cryptocurrency markets with backtesting capabilities.

## Features

- Neural network-based price prediction using PyTorch
- Simple Moving Average (SMA) strategy implementation
- Backtesting framework with performance metrics
- Simulated data generation for strategy testing
- Risk management system

## Project Structure

```
trading_bot/
├── config.py            # Configuration settings
├── data/                # Data storage
├── main.py              # Entry point
├── models/              # ML models
│   ├── torch_price_predictor.py  # PyTorch model
├── strategies/          # Trading strategies
│   ├── simple_moving_average.py  # SMA strategy
│   ├── torch_ml_strategy.py      # PyTorch ML strategy
├── utils/               # Utility functions
│   ├── data_loader.py   # Data acquisition
│   ├── backtester.py    # Backtesting framework
│   ├── visualizations.py # Visualization tools
├── run_bot.py           # Bot execution script
├── run_with_simulated_data.py  # Simulated data testing
└── requirements.txt     # Dependencies
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

## Usage

### Backtesting with Simulated Data

```bash
python run_with_simulated_data.py --strategy torch_ml
```

### Backtesting with Real Data

```bash
python run_bot.py --mode backtest --strategy torch_ml --symbol BTCUSDT --timeframe 1h
```

### Paper Trading

```bash
python run_bot.py --mode paper --strategy torch_ml --symbol BTCUSDT --timeframe 1h
```

### Live Trading (use with caution)

```bash
python run_bot.py --mode live --strategy torch_ml --symbol BTCUSDT --timeframe 1h
```

## Disclaimer

This software is for educational purposes only. Use at your own risk. Cryptocurrency trading involves significant risk and can result in the loss of your invested capital.