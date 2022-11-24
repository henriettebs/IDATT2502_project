#!/bin/sh
#SBATCH --account=share-ie-idi

from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense, LSTM, Dropout, Bidirectional, Layer
from keras import backend as K
from layers.attention import attention
import numpy as np
import tensorflow as tf
from collections import deque
import matplotlib.pyplot as plt

class Lstm:
    
    def Model(self,n_steps,n_features,add_attention):
        model = Sequential()
        if(add_attention):
            model.add(Bidirectional(LSTM(50, return_sequences=True, input_shape=(n_steps, n_features))))
            model.add(attention())
        else: 
            model.add(Bidirectional(LSTM(50, return_sequences=False, input_shape=(n_steps, n_features))))
        model.add(Dropout(0.3))
        model.add(Dense(1))
        opt = Adam(learning_rate=0.0001)
        model.compile(optimizer=opt, loss='mean_squared_error', metrics=['mse'])
        return model

    def split_sequence(self,sequence, n_steps):
        X, y = list(), list()
        for i in range(len(sequence)):
            end_ix = i + n_steps
            if end_ix > len(sequence)-1:
                break
            seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
            X.append(seq_x)
            y.append(seq_y)
        return np.array(X), np.array(y)

def bi_lstm_main(data,pred_days,runs,add_attention):
    n_features = 1
    n_steps = 20
    lstm = Lstm()
    X, y = lstm.split_sequence(data[:-n_steps],n_steps)
    X = X.reshape((X.shape[0], X.shape[1], n_features))

    predictions = [[] for x in range(pred_days)]
    for x in range(runs):
        new_data = data
        # History is loss and mae, loss = how well model predicted values, mae = mean absolute error
        model = lstm.Model(n_steps,n_features,add_attention)
        history  = model.fit(X, y, batch_size=64, epochs=50, verbose=1,validation_split=0.3)
        for x in range(pred_days):
            x_input = np.array(new_data[-20:])
            x_input = x_input.reshape((1, n_steps, n_features))
            pred = model.predict(x_input, verbose=1)
            predictions[x].append(pred[0][0])
            new_data = np.append(new_data,pred)
    avg_result = [np.mean(num_list) for num_list in predictions]
    return avg_result,history