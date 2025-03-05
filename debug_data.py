# debug_data.py - Debug data fetching and processing issues
import pandas as pd
import logging
import os
from datetime import datetime
import matplotlib.pyplot as plt

from utils.data_loader import DataLoader
from models.torch_price_predictor import TorchPricePredictor

# Configure logging
os.makedirs('logs', exist_ok=True)
log_filename = f"logs/debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def debug_data_fetching(symbol='BTCUSDT', timeframe='1h', start_date=None, end_date=None):
    """Debug data fetching issues."""
    logger.info(f"Debugging data fetching for {symbol} ({timeframe})")
    
    # Initialize data loader
    data_loader = DataLoader(symbol=symbol, timeframe=timeframe)
    
    # Fetch data
    data = data_loader.get_historical_data(limit=1000, start_date=start_date, end_date=end_date)
    
    # Check data
    if data.empty:
        logger.error("No data fetched. Check API keys, symbol, and date range.")
        return False
    
    logger.info(f"Successfully fetched {len(data)} data points")
    logger.info(f"Date range: {data.index.min()} to {data.index.max()}")
    logger.info(f"Data columns: {data.columns.tolist()}")
    
    # Check for NaN values
    for col in data.columns:
        nan_count = data[col].isna().sum()
        if nan_count > 0:
            logger.warning(f"Column {col} contains {nan_count} NaN values")
    
    # Plot data
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['close'])
    plt.title(f"{symbol} {timeframe} Price Data")
    plt.grid(True)
    plt.savefig('debug_price_data.png')
    plt.close()
    
    logger.info("Price chart saved to 'debug_price_data.png'")
    
    return data

def debug_feature_creation(data):
    """Debug feature creation issues."""
    logger.info("Debugging feature creation")
    
    # Initialize price predictor
    predictor = TorchPricePredictor()
    
    # Create features
    df_with_features = predictor._create_features(data)
    
    # Check feature data
    logger.info(f"Original data shape: {data.shape}")
    logger.info(f"Feature data shape: {df_with_features.shape}")
    
    # Check for NaN values in features
    for col in df_with_features.columns:
        nan_count = df_with_features[col].isna().sum()
        if nan_count > 0:
            logger.warning(f"Feature {col} contains {nan_count} NaN values")
    
    # Save sample of feature data
    sample = df_with_features.head(20)
    sample.to_csv('debug_features_sample.csv')
    logger.info("Feature sample saved to 'debug_features_sample.csv'")
    
    return df_with_features

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Debug data fetching and processing')
    parser.add_argument('--symbol', default='BTCUSDT', help='Trading symbol/pair')
    parser.add_argument('--timeframe', default='1h', help='Candlestick timeframe')
    parser.add_argument('--start-date', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', help='End date (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    # Debug data fetching
    data = debug_data_fetching(
        symbol=args.symbol,
        timeframe=args.timeframe,
        start_date=args.start_date,
        end_date=args.end_date
    )
    
    if not isinstance(data, pd.DataFrame) or data.empty:
        exit(1)
    
    # Debug feature creation
    features = debug_feature_creation(data)
    
    logger.info("Debugging complete")