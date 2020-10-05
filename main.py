import configme as config
from data import *
from model import *
from predict import *

def main():
    time_window = 'daily'
    for stock in config.stocks:
        print('Working on {}'.format(stock['symbol']))
        save_dataset(stock['symbol'], stock['time_window'])
        create_model(stock['symbol'], stock['time_window'], stock['recent'])
        predict(stock['symbol'], stock['time_window'], stock['recent'])

if __name__ == '__main__':
    # Execute only if run as a script.
    main()