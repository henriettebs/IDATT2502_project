#!/bin/sh
#SBATCH --account=share-ie-idi

from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Bidirectional
from keras.optimizers import Adam
import numpy as np
import time as tm
import datetime as dt
import tensorflow as tf
from yahoo_fin import stock_info as yf
from sklearn.preprocessing import MinMaxScaler
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Bidirectional
from keras.losses import Huber
import matplotlib.pyplot as plt
from keras import backend as K
from keras.layers import Layer


class attention(Layer):
    def __init__(self, return_sequences=True):
        self.return_sequences = return_sequences
        super(attention,self).__init__()

    def build(self, input_shape):
        self.W=self.add_weight(name="att_weight", shape=(input_shape[-1],1),
                               initializer="normal")
        self.b=self.add_weight(name="att_bias", shape=(input_shape[1],1),
                               initializer="normal")
        super(attention,self).build(input_shape)


    def call(self, x):
        e = K.tanh(K.dot(x,self.W)+self.b)
        a = K.softmax(e, axis=1)
        output = x*a
        if self.return_sequences:
            return output
        return K.sum(output, axis=1)



class Lstm:

    def __init__(self):
        print('started')

    def Model(self,n_steps,n_features):
        model = Sequential()
        model.add(LSTM(60, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
        model.add(Dropout(0.5))
        model.add(attention())
        model.add(LSTM(120, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(20))
        model.add(Dense(1))
        opt = Adam(learning_rate=0.0001)
        model.compile(optimizer=opt, loss='mse', metrics=['mse'])
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
    date_3_years_back = ((dt.date.today() - dt.timedelta(days=1)) - dt.timedelta(days=1050)).strftime('%Y-%m-%d')
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

def plotCombined(data,predictions):
    for x in predictions:
        data = np.append(data,x)
    plt.style.use(style='seaborn-whitegrid')
    plt.figure(figsize=(16,10))
    plt.plot(data[-200:])
    plt.xlabel("days")
    plt.ylabel("price")
    plt.legend([f'halo'])
    plt.show()



def main():
    stock = 'KAHOT.OL'
    n_features = 1
    n_steps = 7

    lstm = Lstm()
    scaler = MinMaxScaler()


    data = getCleanData(stock)
    data_visualisation = data.copy()
    data['close'] = scaler.fit_transform(np.expand_dims(data['close'].values, axis=1))



    raw_seq = data['close']
    X, y = lstm.SplitSequence(raw_seq,n_steps)

    X = X.reshape((X.shape[0], X.shape[1], n_features))
    
    model = lstm.Model(n_steps,n_features)


    # History is loss and mae, loss = how well model predicted values, mae = mean absolute error
    history  = model.fit(X, y, batch_size=64, epochs=100, verbose=1,validation_split=0.2)
    x_input = array(data['close'][-7:])
    x_input = x_input.reshape((1, n_steps, n_features))
    yhat = model.predict(x_input, verbose=1)

    
    # plt.figure(figsize=(10, 6))
    # plt.plot(history.history['mse'], label='mse')
    # plt.plot(history.history['loss'], label='loss')
    # plt.legend()
    # plt.show()


    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title('model train vs validation loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'validation'], loc='upper right')
    # plt.show()

    
    print(yhat)

    scaled_yhat = scaler.inverse_transform(yhat)[0][0]
    print(scaled_yhat)

    f = open("myfile.txt", "a")
    f.write(str(scaled_yhat))
    f.close()

    plotCombined(data_visualisation['close'],[scaled_yhat])

    # p = array(data['close'])
    # p = p.reshape((1,len(p),1))

    # forecast = model.predict(p)
    # print(forecast)
    # actual = data['close']

    # plt.figure(figsize=(10, 6))
    # plt.plot(forecast, label="predicted")
    # plt.plot(actual, label="actual")
    # plt.xlabel("Timestep")
    # plt.ylabel("Value")
    # plt.legend()
    # plt.show()




main()