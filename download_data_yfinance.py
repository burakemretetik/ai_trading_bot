# download_data_yfinance.py - Download historical crypto data
import pandas as pd
import os
import logging
from datetime import datetime
import yfinance as yf

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger(__name__)

def download_crypto_data(symbol='BTC-USD', start_date='2023-01-01', end_date='2023-12-31', output_dir='data', interval='1h'):
    """
    Download historical cryptocurrency data using yfinance.
    
    Args:
        symbol (str): Yahoo Finance symbol (e.g., 'BTC-USD', 'ETH-USD')
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format
        output_dir (str): Directory to save the data
        interval (str): Data interval ('1h', '1d', etc.)
    
    Returns:
        str: Path to the saved file
    """
    logger.info(f"Downloading {symbol} data from {start_date} to {end_date} with interval {interval}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Map our interval to yfinance format
    interval_map = {
        '1m': '1m',
        '5m': '5m', 
        '15m': '15m',
        '30m': '30m',
        '1h': '1h',
        '4h': '4h',
        '1d': '1d',
        '1w': '1wk',
        '1mo': '1mo'
    }
    
    yf_interval = interval_map.get(interval, '1h')
    
    try:
        # Download data from Yahoo Finance
        data = yf.download(
            symbol,
            start=start_date,
            end=end_date,
            interval=yf_interval,
            progress=False
        )
        
        if data.empty:
            logger.error(f"No data returned for {symbol}")
            return None
        
        # Process data to match our format
        # Yahoo returns OHLCV data with columns: Open, High, Low, Close, Adj Close, Volume
        data = data.rename(columns={
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        })
        
        # Drop 'Adj Close' column
        if 'Adj Close' in data.columns:
            data = data.drop('Adj Close', axis=1)
        
        # Add a 'timestamp' column if index is not already timestamp
        if not isinstance(data.index, pd.DatetimeIndex):
            data['timestamp'] = pd.to_datetime(data.index)
            data = data.set_index('timestamp')
        
        # Format symbol name for filename
        formatted_symbol = symbol.replace('/', '_').replace('-', '_')
        output_file = f"{formatted_symbol}_{interval}.csv"
        output_path = os.path.join(output_dir, output_file)
        
        # Save to CSV
        data.to_csv(output_path)
        
        logger.info(f"Downloaded {len(data)} records for {symbol}")
        logger.info(f"Date range: {data.index.min()} to {data.index.max()}")
        logger.info(f"Data saved to {output_path}")
        
        return output_path
        
    except Exception as e:
        logger.error(f"Error downloading data: {e}")
        return None

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Download cryptocurrency historical data')
    parser.add_argument('--symbol', default='BTC-USD', help='Yahoo Finance symbol (e.g., BTC-USD, ETH-USD)')
    parser.add_argument('--start-date', default='2023-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', default='2023-12-31', help='End date (YYYY-MM-DD)')
    parser.add_argument('--interval', default='1h', choices=['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w', '1mo'], 
                        help='Data interval')
    parser.add_argument('--output-dir', default='data', help='Output directory')
    
    args = parser.parse_args()
    
    download_crypto_data(
        symbol=args.symbol,
        start_date=args.start_date,
        end_date=args.end_date,
        output_dir=args.output_dir,
        interval=args.interval
    )