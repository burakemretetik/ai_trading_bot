# Installation Guide

This document provides detailed installation instructions for the AI Trading Bot, including handling different Python versions and potential compatibility issues.

## Prerequisites

- Python 3.8-3.12 (Python 3.10 recommended for best compatibility)
- Git
- pip (Python package installer)

## Basic Installation

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

4. Install dependencies based on your Python version:

### For Python 3.8-3.11 (Recommended)

```bash
pip install -r requirements.txt
```

### For Python 3.12

```bash
pip install setuptools wheel
pip install -r requirements_py312.txt --prefer-binary
```

## Troubleshooting Installation Issues

### Common Issues with Python 3.12

Python 3.12 removed the `distutils` module which causes issues with some packages. If you encounter errors related to missing `distutils`, try:

```bash
pip install setuptools wheel
```

Then retry the installation with:

```bash
pip install -r requirements_py312.txt --prefer-binary
```

### Issues with Package Building

If you're experiencing issues with packages that need to be built from source:

1. Install build tools for your operating system:
   - Windows: Microsoft Visual C++ Build Tools
   - Linux: `build-essential`, `python3-dev`
   - macOS: Command Line Tools for Xcode

2. Try installing packages one by one:
```bash
pip install numpy pandas matplotlib
pip install pyyaml python-dotenv
pip install ccxt yfinance
pip install torch scikit-learn joblib
pip install pytest black pytest-cov
```

### Using Conda (Alternative Approach)

If pip installation is problematic, consider using Conda:

```bash
conda create -n trading_env python=3.10
conda activate trading_env
pip install -r requirements.txt
```

## Verifying Installation

After installation, run this command to verify that key dependencies are installed correctly:

```bash
python -c "import numpy, pandas, matplotlib, yaml, torch, sklearn; print('All key dependencies imported successfully!')"
```

## Configuration

1. Create a `.env` file with your API credentials:
```
API_KEY=your_exchange_api_key
API_SECRET=your_exchange_api_secret
```

2. Create necessary directories if they don't exist:
```bash
mkdir -p config data results logs
```

3. Run the bot in simulated mode to verify everything works:
```bash
python main.py --mode simulated --strategy sma --symbol BTCUSDT --timeframe 1h
```