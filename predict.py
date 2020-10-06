import numpy as np
from keras.models import load_model
from util import *
import datetime

def predict(symbol, time_window, recent = None, plot_chart = False):
    model_file = get_model_file(symbol, time_window)
    model = load_model(model_file)

    ohlcv_histories, technical_indicators, next_day_open_values, unscaled_y, y_normaliser,dates = csv_to_dataset(get_data_file(symbol, time_window), recent)

    test_split = 0.9
    n = int(ohlcv_histories.shape[0] * test_split)

    ohlcv_train = ohlcv_histories[:n]
    tech_ind_train = technical_indicators[:n]
    y_train = next_day_open_values[:n]

    ohlcv_test = ohlcv_histories[n:]
    tech_ind_test = technical_indicators[n:]
    y_test = next_day_open_values[n:]
    dates = dates[n-1:]

    unscaled_y_test = unscaled_y[n:]

    y_test_predicted = model.predict([ohlcv_test, tech_ind_test])
    y_test_predicted = y_normaliser.inverse_transform(y_test_predicted)

    buys = []
    sells = []
    
    start = 0
    # end = -1

    x,decisions,i = -1,[],0
    for ohlcv, ind in zip(ohlcv_test[start:], tech_ind_test[start:]):
        normalised_price_today = ohlcv[-1][0]
        normalised_price_today = np.array([[normalised_price_today]]) # Normalized price.
        price_today = y_normaliser.inverse_transform(normalised_price_today) # Original price.
        predicted_price_tomorrow = np.squeeze(y_normaliser.inverse_transform(model.predict([np.array([ohlcv]), np.array([ind])]))) # Fix "Data cardinality is ambiguous" issue: https://github.com/yacoubb/stock-trading-ml/issues/18
        delta = predicted_price_tomorrow - price_today
        thresh = 0.005*price_today
        if delta > thresh:
            buys.append((x, price_today[0][0]))
            decisions.append(('buy',price_today[0][0],dates[i]))
        elif delta < -thresh:
            sells.append((x, price_today[0][0]))
            decisions.append(('sell',price_today[0][0],dates[i]))
        else:
            decisions.append(('hold',price_today[0][0],dates[i]))
        x += 1
        i += 1
    print(f"buys: {len(buys)}")
    print(f"sells: {len(sells)}")
    def compute_earnings(buys_, sells_):
        purchase_amt,stock,balance,last_price,i,j,m,n,max_invest = 1000,0,0,0,0,0,len(buys_),len(sells_),0
        while i < m or j < n:
            if j == n or (i < m and buys_[i][0] < sells_[j][0]):
                # time to buy $10 worth of stock
                last_price = buys_[i][1]
                balance -= purchase_amt
                stock += purchase_amt / last_price
                i += 1
            else:
                # time to sell all of our stock
                last_price = sells_[j][1]
                balance += stock * last_price
                stock = 0
                j += 1
            if balance < 0: max_invest = max(-balance,max_invest)
        print("max invested: {}, earnings: ${}, ${} (closed), ${} (open, {} shares)".format(round(max_invest,2), round(balance+stock*last_price,2), round(balance,2), round(stock*last_price,2), round(stock,2)))
        return round(max_invest,2), round(balance+stock*last_price,2), round(balance,2), round(stock*last_price,2), round(stock,2)

    # we create new lists so we dont modify the original
    max_invest, ret, balance, position, shares = compute_earnings([b for b in buys], [s for s in sells])

    if plot_chart:
        import matplotlib.pyplot as plt
        plt.gcf().set_size_inches(22, 15, forward=True)

        real = plt.plot(unscaled_y_test[start:end], label='real')
        pred = plt.plot(y_test_predicted[start:end], label='predicted')

        if len(buys) > 0:
            plt.scatter(list(list(zip(*buys))[0]), list(list(zip(*buys))[1]), c='#00ff00', s=50)
        if len(sells) > 0:
            plt.scatter(list(list(zip(*sells))[0]), list(list(zip(*sells))[1]), c='#ff0000', s=50)

        # real = plt.plot(unscaled_y[start:end], label='real')
        # pred = plt.plot(y_predicted[start:end], label='predicted')

        plt.legend(['Real', 'Predicted', 'Buy', 'Sell'])
        plt.show()

    result = {'max_invest': max_invest, 'earning': ret, 'earning_rate': round(ret/max_invest,3) if max_invest > 0 else 0, 'buys': len(buys), 'sells': len(sells), 'decisions': decisions, 'time': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    save_predict_result(symbol, time_window, result)
    
