#credit: https://pythonprogramming.net/combining-stock-prices-into-one-dataframe-python-programming-for-finance/

import bs4 as bs
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import os
import pandas as pd
from pandas_datareader import data as pdr
import pickle
import requests
import yfinance as yf

# minor fix of the original code
yf.pdr_override()
# c
TICKERS2 = ["GIS", "NKE", "GS", "IBM","AAPL", 
            "ETN", "FLT", "KO", "HST", "LRCX"]

TICKERS = ['BTC-USD', 'ETH-USD', 'USDT-USD', 'XRP-USD', 'LINK-USD', 'LTC-USD', 'BCH-USD']

def save_sp500_tickers(tickers=False):
    if tickers:
        # self defined tickers
        tickers = TICKERS
    else:
        # download all tickers
        resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        soup = bs.BeautifulSoup(resp.text, 'lxml')
        table = soup.find('table', {'class': 'wikitable sortable'})
        tickers = []
        for row in table.findAll('tr')[1:]:
            ticker = row.findAll('td')[0].text.replace('.', '-')
            ticker = ticker[:-1]
            tickers.append(ticker)
    print(tickers)
    with open("data/sp500tickers.pickle", "wb") as f:
        pickle.dump(tickers, f)
    return tickers


# save_sp500_tickers()
def get_data_from_yahoo(reload_sp500=False):
    if reload_sp500:
        tickers = save_sp500_tickers()
    else:
        with open("data/sp500tickers.pickle", "rb") as f:
            tickers = pickle.load(f)
    if not os.path.exists('data/stock_dfs'):
        os.makedirs('data/stock_dfs')
    start = dt.datetime(2010, 1, 4)
    end = dt.datetime.now()
    for ticker in tickers:
        print(ticker)
        if not os.path.exists('data/stock_dfs/{}.csv'.format(ticker)):
            df = pdr.get_data_yahoo(ticker, start, end)
            df.reset_index(inplace=True)
            df.set_index("Date", inplace=True)
            df.to_csv('data/stock_dfs/{}.csv'.format(ticker))
        else:
            print('Already have {}'.format(ticker))

def compile_data():
    # turn all the dataframes into csv files
    with open("data/sp500tickers.pickle", "rb") as f:
        tickers = pickle.load(f)

    main_df_adj = pd.DataFrame()
    main_df_open = pd.DataFrame()
    main_df_close = pd.DataFrame()
    main_df_high = pd.DataFrame()
    main_df_low = pd.DataFrame()
    main_df_volume = pd.DataFrame()

    for count, ticker in enumerate(tickers):
        df = pd.read_csv('data/stock_dfs/{}.csv'.format(ticker))
        df.set_index('Date', inplace=True)

        df_adj = df.rename(columns={'Adj Close': ticker})
        df_adj.drop(['Open', 'High', 'Low', 'Close', 'Volume'], 1, inplace=True)
        df_open = df.rename(columns={'Open': ticker})
        df_open.drop(['Adj Close', 'High', 'Low', 'Close', 'Volume'], 1, inplace=True)
        df_high = df.rename(columns={'High': ticker})
        df_high.drop(['Open', 'Adj Close', 'Low', 'Close', 'Volume'], 1, inplace=True)
        df_low = df.rename(columns={'Low': ticker})
        df_low.drop(['Open', 'High', 'Adj Close', 'Close', 'Volume'], 1, inplace=True)
        df_close = df.rename(columns={'Close': ticker})
        df_close.drop(['Open', 'High', 'Low', 'Adj Close', 'Volume'], 1, inplace=True)
        df_volume = df.rename(columns={'Volume': ticker})
        df_volume.drop(['Open', 'High', 'Low', 'Close', 'Adj Close'], 1, inplace=True)

        if main_df_adj.empty:
            main_df_adj = df_adj
            main_df_open = df_open
            main_df_close = df_close
            main_df_high = df_high
            main_df_low = df_low
            main_df_volume = df_volume
        else:
            main_df_adj = main_df_adj.join(df_adj, how='outer')
            main_df_open = main_df_open.join(df_open, how='outer')
            main_df_close = main_df_close.join(df_close, how='outer')
            main_df_high = main_df_high.join(df_high, how='outer')
            main_df_low = main_df_low.join(df_low, how='outer')
            main_df_volume = main_df_volume.join(df_volume, how='outer')
        

        if count % 10 == 0:
            print(count)
    print(main_df_adj.head())
    main_df_adj.to_csv('data/sp500_joined_adj.csv')
    main_df_open.to_csv('data/sp500_joined_open.csv')
    main_df_close.to_csv('data/sp500_joined_close.csv')
    main_df_high.to_csv('data/sp500_joined_high.csv')
    main_df_low.to_csv('data/sp500_joined_low.csv')
    main_df_volume.to_csv('data/sp500_joined_volume.csv')


if __name__ == "__main__":
    save_sp500_tickers(tickers=True)
    get_data_from_yahoo(reload_sp500=False)
    compile_data()


