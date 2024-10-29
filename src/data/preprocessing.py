import pandas as pd
import numpy as np
from pathlib import Path
import tulipy as ti
import yfinance as yf
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self):
        self.base_dir = Path('data')
        self.raw_dir = self.base_dir / 'raw'
        self.processed_dir = self.base_dir / 'processed'
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def load_data(self, filename):
        """Load data from CSV file"""
        try:
            filepath = self.raw_dir / filename
            if not filepath.exists():
                logger.error(f"File not found: {filepath}")
                return pd.DataFrame()
                
            df = pd.read_csv(filepath, index_col=0, parse_dates=True)
            logger.info(f"Loaded data shape: {df.shape}")
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

    def process_stock_data(self, symbol):
        """Process stock data for a given symbol"""
        try:
            filename = f"{symbol.replace('.NS', '')}_stock_data.csv"
            df = self.load_data(filename)
            
            if df.empty:
                logger.warning(f"Empty dataframe for {symbol}, skipping processing")
                return

            df = self.handle_missing_values(df)
            df = self.handle_outliers(df, ['Open', 'High', 'Low', 'Close', 'Volume'])
            df = self.add_technical_indicators(df)
            df = self.add_volatility_measures(df)

            # Save processed data
            output_filename = f"{symbol.replace('.NS', '')}_processed.csv"
            df.to_csv(self.processed_dir / output_filename)
            logger.info(f"Processed data saved for {symbol}")
            
        except Exception as e:
            logger.error(f"Error processing stock data for {symbol}: {e}")

    def process_index_data(self, index_name, index_symbol):
        """Process index data"""
        try:
            filename = f"{index_name.replace(' ', '_')}_index_data.csv"
            df = self.load_data(filename)
            
            if df.empty:
                logger. warning(f"Empty dataframe for {index_name}, skipping processing")
                return

            df = self.handle_missing_values(df)
            df = self.handle_outliers(df, ['Open', 'High', 'Low', 'Close', 'Volume'])
            df = self.add_technical_indicators(df)
            df = self.add_volatility_measures(df)

            # Save processed data
            output_filename = f"{index_name.replace(' ', '_')}_processed.csv"
            df.to_csv(self.processed_dir / output_filename)
            logger.info(f"Processed data saved for {index_name}")
            
        except Exception as e:
            logger.error(f"Error processing index data for {index_name}: {e}")

    def process_all_data(self):
        """Process all stock and index data"""
        try:
            # Process stock data
            nifty50_symbols = self.get_nifty50_symbols()
            for symbol in nifty50_symbols:
                self.process_stock_data(symbol)

            # Process index data
            indices = {
                'NIFTY 50': '^NSEI',
                'SENSEX': '^BSESN',
                'NIFTY BANK': '^NSEBANK',
                'NIFTY IT': '^CNXIT',
                'NIFTY AUTO': '0P0001PQB7.BO'
            }
            for index_name, index_symbol in indices.items():
                self.process_index_data(index_name, index_symbol)
                
        except Exception as e:
            logger.error(f"Error processing all data: {e}")

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