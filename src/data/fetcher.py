import yfinance as yf
import pandas as pd
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('market_data.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

sectors = [
    'Technology',
    'Financial Services',
    'Healthcare',
    'Consumer Cyclical',
    'Industrials',
    'Communication Services',
    'Consumer Defensive',
    'Energy',
    'Basic Materials',
    'Real Estate',
    'Utilities'
]

# Define class for fetching stock and index data
class MarketDataFetcher:
    def __init__(self):
        # Directory structure for storing data
        self.base_dir = Path('data')
        self.raw_dir = self.base_dir / 'raw'
        self.indices_dir = self.raw_dir / 'indices'
        self.stocks_dir = self.raw_dir / 'stocks'
        
        # Create necessary directories
        for directory in [self.raw_dir, self.indices_dir, self.stocks_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Known Yahoo Finance symbols for Indian indices
        self.index_symbols = {
            'NIFTY 50': '^NSEI',
            'NIFTY AUTO': '^CNXAUTO',
            'SENSEX': '^BSESN',
            'NIFTY Bank': '^NSEBANK',
            'NIFTY Consumer Durables': '^CNXCONSUMER',
            'NIFTY Financial Services': '^CNXFINANCE',
            'NIFTY FMCG': '^CNXFMCG',
            'NIFTY Healthcare': '^CNXHEALTHCARE',
            'NIFTY IT': '^CNXIT',
            'NIFTY Media': '^CNXMEDIA',
            'NIFTY Metal': '^CNXMETAL',
            'NIFTY Oil & Gas': '^CNXENERGY',
            'NIFTY Pharma': '^CNXPHARMA',
            'NIFTY Private Bank': '^CNXPRIVATBANK',
            'NIFTY PSU Bank': '^CNXPSU',
            'NIFTY Realty': '^CNXREALTY'
        }

    def get_nifty50_symbols(self):
        """Fetch Nifty 50 company symbols from Wikipedia"""
        try:
            nifty50 = pd.read_html('https://en.wikipedia.org/wiki/NIFTY_50')[1]
            symbols = nifty50['Symbol'].tolist()
            symbols = [f"{symbol}.NS" for symbol in symbols]  # Yahoo Finance format
            logger.info(f"Fetched {len(symbols)} Nifty 50 symbols")
            return symbols
        except Exception as e:
            logger.error(f"Error fetching Nifty 50 symbols: {e}")
            return []

    def fetch_data(self, symbol, start_date, end_date):
        """Fetch historical data for a symbol with retries"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                df = yf.Ticker(symbol).history(start=start_date, end=end_date)
                if not df.empty:
                    df.index = df.index.date 
                    return df
                else:
                    logger.warning(f"No data for {symbol}")
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for {symbol}: {e}")
                time.sleep(2**attempt)  # exponential backoff
        logger.error(f"Failed to fetch data for {symbol}")
        return None

    def save_data(self, df, filepath):
        """Save data to a CSV file"""
        try:
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            df.to_csv(filepath)
            logger.info(f"Saved data to {filepath}")
        except Exception as e:
            logger.error(f"Error saving data to {filepath}: {e}")

    def process_symbol(self, symbol, start_date, end_date, is_index=False):
        """Process and save data for a symbol (stock or index)"""
        symbol_type = 'index' if is_index else 'stock'
        try:
            logger.info(f"Processing {symbol_type}: {symbol}")
            df = self.fetch_data(symbol, start_date, end_date)
            if df is not None:
                # Build appropriate filepath for saving data
                filename = f"{symbol.replace('^', '').replace('.NS', '')}_{symbol_type}_data.csv"
                dir_path = self.indices_dir if is_index else self.stocks_dir
                self.save_data(df, dir_path / filename)
                return True
            return False
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
            return False

    def fetch_and_save_all_data(self, start_date, end_date, max_workers=4):
        """Fetch and save data for all Nifty 50 stocks and indices in parallel"""
        # Fetch stock symbols (Nifty 50)
        stock_symbols = self.get_nifty50_symbols()
        
        # Initialize counters for success/failure
        successful_downloads, failed_downloads = 0, 0
        
        # Use ThreadPoolExecutor for concurrent fetching
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # for Nifty 50 stocks
            stock_futures = {
                executor.submit(self.process_symbol, symbol, start_date, end_date): symbol
                for symbol in stock_symbols
            }
            # for Indian indices
            index_futures = {
                executor.submit(self.process_symbol, symbol, start_date, end_date, True): symbol
                for symbol in self.index_symbols.values()
            }
            
            # results
            for future in as_completed({**stock_futures, **index_futures}):
                try:
                    if future.result():
                        successful_downloads += 1
                    else:
                        failed_downloads += 1
                except Exception as e:
                    logger.error(f"Error during download: {e}")
                    failed_downloads += 1
        
        logger.info(f"Download complete. Successful: {successful_downloads}, Failed: {failed_downloads}")

def main():
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365)  #lats 5 years of data
    
    # data fetching
    market_data_fetcher = MarketDataFetcher()
    market_data_fetcher.fetch_and_save_all_data(start_date, end_date)

if __name__ == "__main__":
    main()
