import pandas as pd

def compile_data():
    # turn all the dataframes into csv files

    df = pd.read_csv('data/oil_data/BrentOilPrices.csv')
    df.drop('Date', 1, inplace=True)

    df.to_csv('data/oil_data/oil_price.csv')


compile_data()