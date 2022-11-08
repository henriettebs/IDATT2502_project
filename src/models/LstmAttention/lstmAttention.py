#!/bin/sh
#SBATCH --account=share-ie-idi

from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense, LSTM, Dropout, Bidirectional, Layer
from keras import backend as K

import numpy as np
import tensorflow as tf

from collections import deque
import matplotlib.pyplot as plt


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
        return np.array(X), np.array(y)

# def plotCombined(data,predictions):
#     for x in predictions:
#         data = np.append(data,x)
#     plt.style.use(style='seaborn-whitegrid')
#     plt.figure(figsize=(16,10))
#     plt.plot(data[-200:])
#     plt.xlabel("days")
#     plt.ylabel("price")
#     plt.legend([f'halo'])
#     plt.show()



def lstmAttentionMain(data,pred_days,runs):
    n_features = 1
    n_steps = 7

    lstm = Lstm()

    X, y = lstm.SplitSequence(data,n_steps)
    X = X.reshape((X.shape[0], X.shape[1], n_features))

    predictions = [[] for x in range(pred_days)]
    for x in range(runs):
        new_data = data
        # History is loss and mae, loss = how well model predicted values, mae = mean absolute error
        model = lstm.Model(n_steps,n_features)
        history  = model.fit(X, y, batch_size=64, epochs=100, verbose=1,validation_split=0.2)
        for x in range(pred_days):
            x_input = np.array(new_data[-7:])
            x_input = x_input.reshape((1, n_steps, n_features))
            pred = model.predict(x_input, verbose=1)
            predictions[x].append(pred[0][0])
            new_data = np.append(new_data,pred)
    avg_result = [np.mean(num_list) for num_list in predictions]
    return avg_result