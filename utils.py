from sandbox_secret import SECRET_KEY as SAND_SECRET_KEY
from secret import SECRET_KEY
import os
import pandas as pd
import requests
import json

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
    path = f'stock/{symbol}/chart/{timeframe}?&&token={KEY}'
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
    # read a file from the data folder, pass in the name of the file to the function
    df = pd.read_csv('data\\' + somefile + '.csv')
    # drop unnecessary columns
    df.drop(columns=['Unnamed: 0','symbol','id','key','subkey','label','marketChangeOverTime'], inplace=True)
    return df
