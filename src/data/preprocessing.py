import pandas as pd
import numpy as np
from pathlib import Path
import tulipy as ti
import yfinance as yf
import logging
from datetime import datetime
import threading
import concurrent.futures
import multiprocessing
from sklearn.preprocessing import MinMaxScaler

#logging
log_dir = Path('log')
log_dir.mkdir(parents=True, exist_ok=True)
log_filename = f'preprocessing_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
log_file_path = log_dir / log_filename
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self):
        self.base_dir = Path('data')
        self.raw_dir = self.base_dir / 'raw'
        self.indices_dir = self.raw_dir / 'indices'
        self.stocks_dir = self.raw_dir / 'stocks'
        self.processed_dir = self.base_dir / 'processed'
        self.processed_stocks_dir = self.processed_dir / 'stocks'
        self.processed_indices_dir = self.processed_dir / 'indices'
        
        # Initialize counters
        self.successful_processed = 0
        self.failed_processed = 0
        
        # Create necessary directories
        for directory in [self.processed_dir, self.processed_stocks_dir, self.processed_indices_dir]:
            directory.mkdir(parents=True, exist_ok=True)

    def load_stock_data(self, filename):
        """Load stock data from CSV file"""
        try:
            filepath = self.stocks_dir / filename
            if not filepath.exists():
                logger.error(f"File not found: {filepath}")
                return pd.DataFrame()
                
            df = pd.read_csv(filepath, index_col=0, parse_dates=True)
            logger.info(f"Loaded stock data shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error reading {filepath}: {e}")
            return pd.DataFrame()

    def load_index_data(self, filename):
        """Load index data from CSV file"""
        try:
            filepath = self.indices_dir / filename
            if not filepath.exists():
                logger.error(f"File not found: {filepath}")
                return pd.DataFrame()
                
            df = pd.read_csv(filepath, index_col=0, parse_dates=True)
            logger.info(f"Loaded index data shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error reading {filepath}: {e}")
            return pd.DataFrame()


    def handle_missing_values(self, df):
        """Handle missing values in the dataframe"""
        if df.empty:
            return df
            
        missing_before = df.isnull().sum().sum()
        df = df.ffill()
        df = df.bfill()
        missing_after = df.isnull().sum().sum()
        logger.info(f"Handled missing values: {missing_before - missing_after} values filled")
        return df

    def handle_outliers(self, df, columns, n_std=3):
        """Handle outliers using z-score method"""
        if df.empty:
            return df
            
        for col in columns:
            if col not in df.columns:
                logger.warning(f"Column {col} not found in dataframe")
                continue
                
            mean = df[col].mean()
            std = df[col].std()
            outliers_before = df[np.abs(df[col] - mean) > n_std * std].shape[0]
            df[col] = df[col].clip(lower=mean - n_std*std, upper=mean + n_std*std)
            outliers_after = df[np.abs(df[col] - mean) > n_std * std].shape[0]
            logger.info(f"Handled {outliers_before - outliers_after} outliers in {col}")
        return df

    def add_technical_indicators(self, df):
        """Add technical indicators to the dataframe"""
        if df.empty:
            return df
            
        try:
            # Convert to numpy arrays and ensure float64 dtype
            close = df['Close'].astype(float).to_numpy()
            high = df['High'].astype(float).to_numpy()
            low = df['Low'].astype(float).to_numpy()

            # Simple Moving Average
            try:
                df['SMA_10'] = pd.Series(ti.sma(close, period=10), index=df.index[-len(close)+9:])
                df['SMA_50'] = pd.Series(ti.sma(close, period=50), index=df.index[-len(close)+49:])
            except Exception as e:
                logger.error(f"Error calculating SMA: {e}")

            # Exponential Moving Average
            try:
                df['EMA_10'] = pd.Series(ti.ema(close, period=10), index=df.index[-len(close)+9:])
                df['EMA_50'] = pd.Series(ti.ema(close, period=50), index=df.index[-len(close)+49:])
            except Exception as e:
                logger.error(f"Error calculating EMA: {e}")

            # Relative Strength Index
            try:
                df['RSI'] = pd.Series(ti.rsi(close, period=14), index=df.index[-len(close)+13:])
            except Exception as e:
                logger.error(f"Error calculating RSI: {e}")

            # Moving Average Convergence Divergence
            try:
                macd, signal, hist = ti.macd(close, short_period=12, long_period=26, signal_period=9)
                df['MACD'] = pd.Series(macd, index=df.index[-len(macd):])
                df['MACD_Signal'] = pd.Series(signal, index=df.index[-len(signal):])
            except Exception as e:
                logger.error(f"Error calculating MACD: {e}")

            # Bollinger Bands
            try:
                upper, middle, lower = ti.bbands(close, period=20, stddev=2)
                df['Upper_BB'] = pd.Series(upper, index=df.index[-len(upper):])
                df['Middle_BB'] = pd.Series(middle, index=df.index[-len(middle):])
                df['Lower_BB'] = pd.Series(lower, index=df.index[-len(lower):])
            except Exception as e:
                logger.error(f"Error calculating Bollinger Bands: {e}")

            logger.info("Technical indicators added successfully")
            return df
            
        except Exception as e:
            logger.error(f"Error in add_technical_indicators: {e}")
            return df

    def add_volatility_measures(self, df):
        """Add volatility measures to the dataframe"""
        if df.empty:
            return df
            
        try:
            # Daily Returns
            df['Daily_Return'] = df['Close'].pct_change()

            # Rolling Volatility (20-day window)
            df['Volatility_20'] = df['Daily_Return'].rolling(window=20).std() * np.sqrt(252)

            # Average True Range
            high = df['High'].astype(float).to_numpy()
            low = df['Low'].astype(float).to_numpy()
            close = df['Close'].astype(float).to_numpy()
            
            try:
                atr = ti.atr(high, low, close, period=14)
                df['ATR'] = pd.Series(atr, index=df.index[-len(atr):])
            except Exception as e:
                logger.error(f"Error calculating ATR: {e}")

            logger.info("Volatility measures added successfully")
            return df
            
        except Exception as e:
            logger.error(f"Error in add_volatility_measures: {e}")
            return df

    def process_all_data(self):
        """Process all stock and index data using parallel processing"""
        start_time = datetime.now()
        logger.info("Starting data preprocessing...")
        
        try:
            # Reset counters
            self.successful_processed = 0
            self.failed_processed = 0
            
            # Get symbols and indices
            nifty50_symbols = self.get_nifty50_symbols()
            indices = {
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
            
            total_symbols = len(nifty50_symbols) + len(indices)
            logger.info(f"Total files to process: {total_symbols}")

            # Determine number of workers
            max_workers = min(multiprocessing.cpu_count(), total_symbols)
            logger.info(f"Using {max_workers} workers for parallel processing")

            # Process stocks and indices in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit stock processing tasks
                stock_futures = {
                    executor.submit(self.process_stock_data, symbol): symbol 
                    for symbol in nifty50_symbols
                }
                
                # Submit index processing tasks
                index_futures = {
                    executor.submit(self.process_index_data, name, symbol): name 
                    for name, symbol in indices.items()
                }

                # Combine all futures
                all_futures = {**stock_futures, **index_futures}

                # Wait for completion and process results
                for future in concurrent.futures.as_completed(all_futures):
                    task_id = all_futures[future]
                    try:
                        future.result()  # This will raise any exceptions that occurred
                        logger.info(f"Completed processing {task_id}")
                    except Exception as e:
                        logger.error(f"Error processing {task_id}: {e}")

            # Log final statistics
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            logger.info("\nPreprocessing Summary:")
            logger.info(f"Total files attempted: {total_symbols}")
            logger.info(f"Successfully processed: {self.successful_processed}")
            logger.info(f"Failed to process: {self.failed_processed}")
            logger.info(f"Success rate: {(self.successful_processed/total_symbols)*100:.2f}%")
            logger.info(f"Total processing time: {processing_time:.2f} seconds")
                
        except Exception as e:
            logger.error(f"Error in parallel processing: {e}")
            
    def scale_ohlc(self, df):
        """Scale Open, High, Low, Close to range 0-1"""
        if df.empty:
            return df

        try:
            scaler = MinMaxScaler()
            cols_to_scale = ['Open', 'High', 'Low', 'Close', 'Volume']
            
            # Check if all columns exist
            if all(col in df.columns for col in cols_to_scale):
                df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
                logger.info("OHLC values scaled to 0-1 range")
            else:
                missing_cols = [col for col in cols_to_scale if col not in df.columns]
                logger.warning(f"Columns {missing_cols} not found. Skipping OHLC scaling.")
            
            return df
        except Exception as e:
            logger.error(f"Error in scale_ohlc: {e}")
            return df

    def process_stock_data(self, symbol):
        try:
            filename = f"{symbol.replace('.NS', '')}_stock_data.csv"
            df = self.load_stock_data(filename)
            
            if df.empty:
                logger.warning(f"Empty dataframe for {symbol}, skipping processing")
                with threading.Lock():
                    self.failed_processed += 1
                return

            df = self.handle_missing_values(df)
            df = self.handle_outliers(df, ['Open', 'High', 'Low', 'Close', 'Volume'])
            df = self.scale_ohlc(df)  # Add this line
            df = self.add_technical_indicators(df)
            df = self.add_volatility_measures(df)

            # Save processed data
            output_filename = f"{symbol.replace('.NS', '')}_processed.csv"
            df.to_csv(self.processed_stocks_dir / output_filename)
            logger.info(f"Processed data saved for {symbol}")
            
            with threading.Lock():
                self.successful_processed += 1
            
        except Exception as e:
            logger.error(f"Error processing stock data for {symbol}: {e}")
            with threading.Lock():
                self.failed_processed += 1

    def process_index_data(self, index_name, index_symbol):
        try:
            filename = f"{index_symbol.replace('^', '')}_index_data.csv"
            df = self.load_index_data(filename)
            
            if df.empty:
                logger.warning(f"Empty dataframe for {index_name}, skipping processing")
                with threading.Lock():
                    self.failed_processed += 1
                return

            df = self.handle_missing_values(df)
            df = self.handle_outliers(df, ['Open', 'High', 'Low', 'Close', 'Volume'])
            df = self.scale_ohlc(df)  # Add this line
            df = self.add_technical_indicators(df)
            df = self.add_volatility_measures(df)

            # Save processed data
            output_filename = f"{index_name.replace(' ', '_')}_processed.csv"
            df.to_csv(self.processed_indices_dir / output_filename)
            logger.info(f"Processed data saved for {index_name}")
            
            with threading.Lock():
                self.successful_processed += 1
            
        except Exception as e:
            logger.error(f"Error processing index data for {index_name}: {e}")
            with threading.Lock():
                self.failed_processed += 1

    def get_nifty50_symbols(self):
        """Get list of Nifty 50 companies"""
        try:
            nifty50 = pd.read_html('https://en.wikipedia.org/wiki/NIFTY_50')[1]
            symbols = nifty50['Symbol'].tolist()
            symbols = [f"{symbol}.NS" for symbol in symbols]
            return symbols
        except Exception as e:
            logger.error(f"Error fetching Nifty 50 symbols: {e}")
            return []

def main():
    preprocessor = DataPreprocessor()
    preprocessor.process_all_data()

if __name__ == "__main__":
    main()