from sandbox_secret import SECRET_KEY as SAND_SECRET_KEY
from secret import SECRET_KEY
from sklearn.preprocessing import StandardScaler

import os
import numpy as np
import pandas as pd
import requests
import json

# retrieve a dataset from the IEX Cloud API
def get_dataset(symbol, timeframe='max', version='sandbox'):
    # Flag to use production api or not. Add your keys in sandbox_secret.py/secret.py files located in the root directory
    if version == 'stable':
        url_prefix = "https://cloud.iexapis.com/stable"
        KEY = SECRET_KEY
    else:
        url_prefix = "https://sandbox.iexapis.com/stable/"
        KEY = SAND_SECRET_KEY

    # Gets the stock price at closing for some date parameter
    # path = f'stock/{symbol}/chart/{timeframe}?chartCloseOnly=True&&token={SECRET_KEY}'
    path = f'stock/{symbol}/chart/{timeframe}?filter=date,close,high,low,open,volume,&&token={KEY}'
    full_url = requests.compat.urljoin(url_prefix, path)

    resp = requests.get(full_url)
    # create json object from response
    prices_obj = json.loads(resp.text)
    # turn json object into dataframe
    prices = pd.DataFrame(prices_obj)

    # make a 'data' directory only if one does not already exist
    os.makedirs('data', exist_ok=True)

    # export data to csv and place in data folder
    prices.to_csv('data\\' + symbol + '.csv')

def load_dataset(somefile):
    # read a file from the data folder, pass in the name of the file to the function.
    # Only looking at closing prices
    df = pd.read_csv('data\\' + somefile + '.csv', usecols=['close'])

    # drop IEX index column
    # df.drop(columns=['Unnamed: 0'], inplace=True)
    
    return np.array([df['close'].values])

def get_scaler(env):
    # taking an env and returns a scaler for the observation space
    low = [0] * (env.n_stock * 2 + 1)
    high = []
    max_price = env.stock_price_history.max(axis=1)
    min_price = env.stock_price_history.min(axis=1)
    max_cash = env.init_invest * 3
    max_stock_owned = max_cash // min_price
    for i in max_stock_owned:
        high.append(i)
    for i in max_price:
        high.append(i)
    high.append(max_cash)

    scaler = StandardScaler()
    scaler.fit([low, high])
    return scaler

def maybe_make_dir(directory):
  if not os.path.exists(directory):
    os.makedirs(directory)