from sandbox_secret import SECRET_KEY
import os
import pandas as pd
import requests
import json

def get_dataset(symbol, timeframe='max'):
    url_prefix = "https://sandbox.iexapis.com/stable/"

    # Gets the stock price at closing for some date parameter
    # path = f'stock/{symbol}/chart/{timeframe}?chartCloseOnly=True&&token={SECRET_KEY}'
    path = f'stock/{symbol}/chart/{timeframe}?&&token={SECRET_KEY}'
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
    pass
