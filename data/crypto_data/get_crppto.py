import pandas as pd

def compile_data():
    # turn all the dataframes into csv files
    tickers = ['BTC', 'ETH', 'LTC', 'BCH', 'TRX', 'XRP', 'EOS']

    main_df_open = pd.DataFrame()
    main_df_close = pd.DataFrame()
    main_df_high = pd.DataFrame()
    main_df_low = pd.DataFrame()
    main_df_volumeto = pd.DataFrame()
    main_df_volumefrom = pd.DataFrame()

    for count, ticker in enumerate(tickers):
        df = pd.read_csv('data/crypto_data/coins/{}.csv'.format(ticker))

        df_open = df.rename(columns={'open': ticker})
        df_open.drop(['volumefrom', 'high', 'low', 'close', 'volumeto'], 1, inplace=True)
        df_high = df.rename(columns={'high': ticker})
        df_high.drop(['open', 'volumefrom', 'low', 'close', 'volumeto'], 1, inplace=True)
        df_low = df.rename(columns={'low': ticker})
        df_low.drop(['open', 'high', 'volumefrom', 'close', 'volumeto'], 1, inplace=True)
        df_close = df.rename(columns={'close': ticker})
        df_close.drop(['open', 'high', 'low', 'volumefrom', 'volumeto'], 1, inplace=True)
        df_volumeto = df.rename(columns={'volumeto': ticker})
        df_volumeto.drop(['open', 'high', 'low', 'close', 'volumefrom'], 1, inplace=True)
        df_volumefrom = df.rename(columns={'volumefrom': ticker})
        df_volumefrom.drop(['open', 'high', 'low', 'close', 'volumeto'], 1, inplace=True)

        if main_df_open.empty:
            main_df_open = df_open
            main_df_close = df_close
            main_df_high = df_high
            main_df_low = df_low
            main_df_volumeto = df_volumeto
            main_df_volumefrom = df_volumefrom
        else:
            main_df_open = main_df_open.join(df_open, how='outer')
            main_df_close = main_df_close.join(df_close, how='outer')
            main_df_high = main_df_high.join(df_high, how='outer')
            main_df_low = main_df_low.join(df_low, how='outer')
            main_df_volumeto = main_df_volumeto.join(df_volumeto, how='outer')
            main_df_volumefrom = main_df_volumefrom.join(df_volumefrom, how='outer')
        

        if count % 10 == 0:
            print(count)
    print(main_df_open.head())
    main_df_open.to_csv('data/crypto_data/coins_joined_open.csv')
    main_df_close.to_csv('data/crypto_data/coins_joined_close.csv')
    main_df_high.to_csv('data/crypto_data/coins_joined_high.csv')
    main_df_low.to_csv('data/crypto_data/coins_joined_low.csv')
    main_df_volumeto.to_csv('data/crypto_data/coins_joined_volumeto.csv')
    main_df_volumefrom.to_csv('data/crypto_data/coins_joined_volumefrom.csv')

compile_data()