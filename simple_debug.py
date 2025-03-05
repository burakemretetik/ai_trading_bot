# fixed_simple_debug.py - Debug script with proper CCXT symbol formatting
import logging
import os
from datetime import datetime
import ccxt
import pandas as pd
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def format_symbol(symbol):
    """Convert concatenated symbol to CCXT format with slash."""
    if '/' in symbol:
        return symbol
    
    # Most common format is BTCUSDT -> BTC/USDT
    if len(symbol) >= 6 and 'USDT' in symbol:
        base = symbol.replace('USDT', '')
        return f"{base}/USDT"
    
    # Try to find other quote currencies (BTC, ETH, etc.)
    for quote in ['BTC', 'ETH', 'BNB', 'USDC']:
        if symbol.endswith(quote):
            base = symbol.replace(quote, '')
            return f"{base}/{quote}"
    
    # Default: just return the original
    return symbol

def get_available_symbols():
    """Get available symbols from the exchange."""
    try:
        exchange = ccxt.binance({'enableRateLimit': True})
        markets = exchange.load_markets()
        symbols = list(markets.keys())
        return symbols
    except Exception as e:
        logger.error(f"Error getting symbols: {e}")
        return []

def test_data_fetching_direct():
    """Test data fetching using CCXT directly."""
    logger.info("Testing direct CCXT data fetching...")
    
    try:
        # Initialize exchange
        exchange = ccxt.binance({'enableRateLimit': True})
        
        # Load markets
        markets = exchange.load_markets()
        
        # Find BTC/USDT pair
        btc_usdt = 'BTC/USDT'
        if btc_usdt in markets:
            logger.info(f"Found {btc_usdt} in available markets")
            
            # Fetch OHLCV data
            ohlcv = exchange.fetch_ohlcv(btc_usdt, '1h', limit=10)
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            logger.info(f"Successfully fetched {len(df)} rows of data")
            logger.info(f"First row: {df.iloc[0].to_dict()}")
            
            return True
        else:
            logger.error(f"{btc_usdt} not found in available markets")
            sample_symbols = list(markets.keys())[:10]
            logger.info(f"Sample available symbols: {sample_symbols}")
            return False
            
    except Exception as e:
        logger.error(f"Error in direct testing: {e}")
        return False

def search_for_symbol(symbol_part):
    """Search for symbols containing the given part."""
    try:
        exchange = ccxt.binance({'enableRateLimit': True})
        markets = exchange.load_markets()
        
        matching_symbols = [s for s in markets.keys() if symbol_part in s]
        
        logger.info(f"Found {len(matching_symbols)} symbols containing '{symbol_part}'")
        if matching_symbols:
            logger.info(f"Sample matching symbols: {matching_symbols[:10]}")
        
        return matching_symbols
    except Exception as e:
        logger.error(f"Error searching for symbols: {e}")
        return []

if __name__ == "__main__":
    logger.info("Starting debug script...")
    
    # Test direct CCXT fetching
    success = test_data_fetching_direct()
    
    if not success:
        # Search for BTC and USDT symbols
        logger.info("Searching for BTC symbols...")
        btc_symbols = search_for_symbol('BTC')
        
        logger.info("Searching for USDT symbols...")
        usdt_symbols = search_for_symbol('USDT')
    
    logger.info("Debug script completed")