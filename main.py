from utils import get_dataset
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--getdata', nargs='+', type=str, help='retrieve data set using IEX cloud API. Takes two args: symbol, timeframe')
    args = parser.parse_args()

    if args.getdata is not None:
        get_dataset(*args.getdata)
    