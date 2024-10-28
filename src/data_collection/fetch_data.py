import yahoofinancials
import pandas as pd
import numpy as np

def fetch_stock_data(ticker, start_date, end_date):
    """
    Fetches historical stock data from Yahoo Finance API.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol.
    start_date : str
        Start date in 'YYYY-MM-DD' format.
    end_date : str
        End date in 'YYYY-MM-DD' format.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame containing historical stock data.
    """
    stock = yahoofinancials.Stock(ticker)
    hist = stock.get_historical_price_data(start_date, end_date, 'daily')
    df = pd.DataFrame(hist['prices'])
    df['date'] = pd.to_datetime(df['date'], unit='ms')
    df.set_index('date', inplace=True)
    df.rename(columns={'adjclose': 'adj_close'}, inplace=True)

    # Add technical indicators
    df['ma_50'] = df['adj_close'].rolling(window=50).mean()
    df['ma_200'] = df['adj_close'].rolling(window=200).mean()
    df['rsi'] = talib.RSI(df['adj_close'], timeperiod=14)
    df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(df['adj_close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(df['adj_close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)

    # Add volume data
    df['volume'] = df['volume'].astype(int)

    # Add sentiment analysis data (optional)
    # ...

    # Add macro-economic data (interest rates, inflation)
    # ...

    # Add company-specific features (P/E ratio, earnings reports)
    # ...

    return df
