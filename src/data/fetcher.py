import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import time
import logging
# from sqlalchemy import create_engine, Column, Integer, Float, String, Date
# from sqlalchemy.ext.declarative import declarative_base
# from sqlalchemy.orm import sessionmaker

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# # SQLAlchemy setup
# Base = declarative_base()

# class StockData(Base):
#     __tablename__ = 'stocks'

#     id = Column(Integer, primary_key=True)
#     date = Column(Date)
#     symbol = Column(String)
#     open = Column(Float)
#     high = Column(Float)
#     low = Column(Float)
#     close = Column(Float)
#     volume = Column(Integer)

# class IndexData(Base):
#     __tablename__ = 'indices'

#     id = Column(Integer, primary_key=True)
#     date = Column(Date)
#     index_name = Column(String)
#     open = Column(Float)
#     high = Column(Float)
#     low = Column(Float)
#     close = Column(Float)
#     volume = Column(Integer)

class IndianMarketData:
    def __init__(self):
        # Create directories for data storage
        self.base_dir = Path('data')
        self.raw_dir = self.base_dir / 'raw'
        self.processed_dir = self.base_dir / 'processed'
        
        # Create directories if they don't exist
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # # database setup
        # self.db_url = 'postgresql://username:password@localhost:5432/indian_market_data'
        # self.engine = create_engine(self.db_url)
        # Base.metadata.create_all(self.engine)
        # self.Session = sessionmaker(bind=self.engine)
        
        # major Indian indices
        self.indices = self.get_index_data()
        self.validate_indices()

    def get_index_data(self):
        """Fetch NSE index data from Wikipedia"""
        try:
            # Fetch tables from Wikipedia page
            url = 'https://en.wikipedia.org/wiki/NSE_Indices'
            tables = pd.read_html(url)
            
            # The main indices table is typically one of the first tables
            # We'll need to process multiple tables to get different types of indices
            indices = {}
            
            # Process Broad Market Indices
            broad_market = tables[1]  # Adjust index based on actual table position
            for _, row in broad_market.iterrows():
                index_name = row['Index Name']
                if pd.notna(index_name):  # Check if index name is not NaN
                    # Convert index name to Yahoo Finance symbol format
                    symbol = self.get_yahoo_finance_symbol(index_name)
                    if symbol:
                        indices[index_name] = symbol
            
            # Process Sectoral Indices
            sectoral = tables[2]  # Adjust index based on actual table position
            for _, row in sectoral.iterrows():
                index_name = row['Index Name']
                if pd.notna(index_name):
                    symbol = self.get_yahoo_finance_symbol(index_name)
                    if symbol:
                        indices[index_name] = symbol
            
            # If no indices were found, use default indices as fallback
            if not indices:
                logger.warning("No indices found from Wikipedia, using default indices")
                indices = {
                    'NIFTY 50': '^NSEI',
                    'SENSEX': '^BSESN',
                    'NIFTY BANK': '^NSEBANK',
                    'NIFTY IT': '^CNXIT',
                    'NIFTY AUTO': '^0P0001PQB7'
                }
            
            logger.info(f"Successfully fetched {len(indices)} indices")
            return indices
        
        except Exception as e:
            logger.error(f"Error fetching index data from Wikipedia: {e}")
            # Return default indices as fallback
            return {
                'NIFTY 50': '^NSEI',
                'SENSEX': '^BSESN',
                'NIFTY BANK': '^NSEBANK',
                'NIFTY IT': '^CNXIT',
                'NIFTY AUTO': '^0P0001PQB7'
            }

    def get_yahoo_finance_symbol(self, index_name):
        """Convert index name to Yahoo Finance symbol"""
        # Mapping of index names to Yahoo Finance symbols
        symbol_mapping = {
            'NIFTY 50': '^NSEI',
            'NIFTY NEXT 50': '^CNXJUNIOR',
            'NIFTY 100': '^CNX100',
            'NIFTY 200': '^CNX200',
            'NIFTY 500': '^CRSLDX',
            'NIFTY BANK': '^NSEBANK',
            'NIFTY IT': '^CNXIT',
            'NIFTY AUTO': '^0P0001PQB7',
            'NIFTY FINANCIAL SERVICES': '^CNXFINANCE',
            'NIFTY FMCG': '^CNXFMCG',
            'NIFTY MEDIA': '^CNXMEDIA',
            'NIFTY METAL': '^CNXMETAL',
            'NIFTY PHARMA': '^CNXPHARMA',
            'NIFTY PSU BANK': '^CNXPSUBANK',
            'NIFTY PRIVATE BANK': '^NIFTYPRBANK',
            'NIFTY REALTY': '^CNXREALTY',
            # Add more mappings as needed
        }
        
        # Try to find exact match
        if index_name in symbol_mapping:
            return symbol_mapping[index_name]
        
        # Try to find case-insensitive match
        index_name_upper = index_name.upper()
        for known_name, symbol in symbol_mapping.items():
            if known_name.upper() == index_name_upper:
                return symbol
        
        logger.warning(f"No Yahoo Finance symbol found for index: {index_name}")
        return None

    def validate_indices(self):
        """Validate that we can fetch data for each index"""
        valid_indices = {}
        for index_name, symbol in self.indices.items():
            if self.is_ticker_valid(symbol):
                valid_indices[index_name] = symbol
            else:
                logger.warning(f"Removing invalid index: {index_name} ({symbol})")
        
        self.indices = valid_indices
        logger.info(f"Validated {len(valid_indices)} indices")
    def get_nifty50_symbols(self):
        """Get list of Nifty 50 companies"""
        try:
            nifty50 = pd.read_html('https://en.wikipedia.org/wiki/NIFTY_50')[1]
            symbols = nifty50['Symbol'].tolist()
            # Add .NS suffix for Yahoo Finance
            symbols = [f"{symbol}.NS" for symbol in symbols]
            return symbols
        except Exception as e:
            logger.error(f"Error fetching Nifty 50 symbols: {e}")
            return []

    def fetch_stock_data(self, symbol, start_date, end_date):
        """Fetch historical data for a given stock"""
        try:
            stock = yf.Ticker(symbol)
            df = stock.history(start=start_date, end=end_date)
            df.index = df.index.date  # Convert timestamp to date
            return df
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None

    def fetch_index_data(self, index_symbol, start_date, end_date):
        """Fetch historical data for indices"""
        try:
            index_data = yf.download(index_symbol, start=start_date, end=end_date)
            if index_data.empty:
                logger.error(f"No data found for index {index_symbol}. It may be delisted.")
                return None
            index_data.index = index_data.index.date
            return index_data
        except Exception as e:
            logger.error(f"Error fetching data for index {index_symbol}: {e}")
            return None

    def save_to_csv(self, df, filename):
        """Save data to CSV file"""
        try:
            filepath = self.raw_dir / filename
            df.to_csv(filepath)
            logger.info(f"Data saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving CSV file {filename}: {e}")

    # def save_to_db(self, df, symbol, table_name):
    #     """Save data to PostgreSQL database"""
    #     try:
    #         session = self.Session()
            
    #         for index, row in df.iterrows():
    #             if table_name == 'stocks':
    #                 data = StockData(
    #                     date=index,
    #                     symbol=symbol,
    #                     open=row['Open'],
    #                     high=row['High'],
    #                     low=row['Low'],
    #                     close=row['Close'],
    #                     volume=row['Volume']
    #                 )
    #             else:  # indices
    #                 data = IndexData(
    #                     date=index,
    #                     index_name=symbol,
    #                     open=row['Open'],
    #                     high=row['High'],
    #                     low=row['Low'],
    #                     close=row['Close'],
    #                     volume=row['Volume']
    #                 )
    #             session.add(data)
            
    #         session.commit()
    #         session.close()
    #         logger.info(f"Data saved to database table {table_name}")
    #     except Exception as e:
    #         logger.error(f"Error saving to database: {e}")
    #         session.rollback()
    #     finally:
    #         session.close()
    
    
    def is_ticker_valid(self, ticker):
        """Check if the ticker is valid by attempting to fetch data"""
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(period="1d")
            return not data.empty
        except Exception as e:
            logger.error(f"Error checking ticker {ticker}: {e}")
            return False

    def fetch_and_save_all_data(self, start_date, end_date):
        """Main function to fetch and save all data"""
        # Fetch and save stock data
        symbols = self.get_nifty50_symbols()
        
        for symbol in symbols:
            logger.info(f"Fetching data for {symbol}")
            df = self.fetch_stock_data(symbol, start_date, end_date)
            
            if df is not None and not df.empty:
                self.save_to_csv(df, f"{symbol.replace('.NS', '')}_stock_data.csv")     #to CSV
                # self.save_to_db(df, symbol, 'stocks')       # to database
            else:
                logger.warning(f"No data available for {symbol}. Skipping.")
            
            time.sleep(1)  # Prevent rate limiting

        # Fetch and save index data
        for index_name, index_symbol in self.indices.items():
            if not self.is_ticker_valid(index_symbol):
                logger.warning(f"Skipping invalid index ticker: {index_symbol}")
                continue
            logger.info(f"Fetching data for index {index_name}")
            df = self.fetch_index_data(index_symbol, start_date, end_date)
            
            if df is not None and not df.empty:
                self.save_to_csv(df, f"{index_name.replace(' ', '_')}_index_data.csv")                      # Save to CSV
                # self.save_to_db(df, index_name, 'indices')      # Save to database
            else:
                logger.warning(f"No data available for index {index_name}. Skipping.")
            
            time.sleep(1) 

def main():
    # Set date range for data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*5)  # 5 years of data
    
    # Initialize and run data collection
    market_data = IndianMarketData()
    market_data.fetch_and_save_all_data(start_date, end_date)

if __name__ == "__main__":
    main()