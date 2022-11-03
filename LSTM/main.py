
from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Bidirectional
import numpy as np
import time as tm
import datetime as dt
import tensorflow as tf
from yahoo_fin import stock_info as yf
from sklearn.preprocessing import MinMaxScaler
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Bidirectional
import matplotlib.pyplot as plt


class Lstm:

    def __init__(self):
        print('started')

    def Model(self,n_steps,n_features):
        model = Sequential()
        model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
        model.add(LSTM(50, activation='relu'))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse',metrics=['mae'])
        return model

    def SplitSequence(self,sequence, n_steps):
        X, y = list(), list()
        for i in range(len(sequence)):
            end_ix = i + n_steps
            if end_ix > len(sequence)-1:
                break
            seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
            X.append(seq_x)
            y.append(seq_y)
        return array(X), array(y)

def getTimeInterval():
    date_now = tm.strftime('%Y-%m-%d')
    date_3_years_back = ((dt.date.today() - dt.timedelta(days=1)) - dt.timedelta(days=50)).strftime('%Y-%m-%d')
    return date_now, date_3_years_back

def scaleData(data):
    scaler = MinMaxScaler()
    data['close'] = scaler.fit_transform(np.expand_dims(data['close'].values, axis=1))
    return data

def getCleanData(stock):
    date_now,date_3_years_back = getTimeInterval()
    init_df = yf.get_data(
    stock, 
    start_date=date_3_years_back, 
    end_date=date_now,

    interval='1d')
    init_df = init_df.drop(['open', 'high', 'low', 'adjclose', 'ticker', 'volume'], axis=1)
    init_df['date'] = init_df.index
    return init_df




def main():
    stock = 'KAHOT.OL'
    n_features = 1
    n_steps = 3

    lstm = Lstm()
    scaler = MinMaxScaler()


    data = getCleanData(stock)
    data['close'] = scaler.fit_transform(np.expand_dims(data['close'].values, axis=1))



    raw_seq = data['close']
    X, y = lstm.SplitSequence(raw_seq,n_steps)

    X = X.reshape((X.shape[0], X.shape[1], n_features))
    
    model = lstm.Model(n_steps,n_features)
    history  = model.fit(X, y, epochs=100, verbose=1)

    plt.figure(figsize=(10, 6))
    plt.plot(history.history['mae'], label='mae')
    plt.plot(history.history['loss'], label='loss')
    plt.legend()
    plt.show()

    x_input = array(data['close'][-3:])
    x_input = x_input.reshape((1, n_steps, n_features))
    yhat = model.predict(x_input, verbose=1)
    print(yhat)

    scaled_yhat = scaler.inverse_transform(yhat)[0][0]
    print(scaled_yhat)


    p = array(data['close'])
    p = p.reshape((1,len(p),1))

    forecast = model.predict(p)
    print(forecast)
    actual = data['close']

    plt.figure(figsize=(10, 6))
    plt.plot(forecast, label="predicted")
    plt.plot(actual, label="actual")
    plt.xlabel("Timestep")
    plt.ylabel("Value")
    plt.legend()
    plt.show()




main()