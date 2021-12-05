from utils import get_dataset, load_dataset
import argparse

if __name__ == '__main__':
    # command line interface
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--getdata', nargs='+', type=str, 
        help='retrieve data set using IEX cloud API. Takes 3 args: str symbol, str timeframe (see IEX Cloud docs for values), str version (using production API or not)')
    args = parser.parse_args()

    if args.getdata is not None:
        # * operator used to expand iterable into function call
        get_dataset(*args.getdata)
    
    tmp = load_dataset('IBM')




    