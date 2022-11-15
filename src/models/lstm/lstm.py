#!/bin/sh
#SBATCH --account=share-ie-idi

from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense, LSTM, Dropout
from keras import backend as K
import numpy as np
import tensorflow as tf
from collections import deque
import matplotlib.pyplot as plt
from layers.attention import attention

class Lstm:
    def Model(self,n_steps,n_features,add_attention):
        model = Sequential()
        model.add(LSTM(60, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
        model.add(Dropout(0.3))
        if(add_attention):
            model.add(attention())
        model.add(LSTM(120, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(1))
        opt = Adam(learning_rate=0.0001)
        model.compile(optimizer=opt, loss='mse', metrics=['mse'])
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

def lstm_main(data,pred_days,runs,add_attention):
    n_features = 1
    n_steps = 7
    lstm = Lstm()
    X, y = lstm.split_sequence(data,n_steps)
    X = X.reshape((X.shape[0], X.shape[1], n_features))

    predictions = [[] for x in range(pred_days)]
    for x in range(runs):
        new_data = data
        # History is loss and mae, loss = how well model predicted values, mae = mean absolute error
        model = lstm.Model(n_steps,n_features,add_attention)
        history  = model.fit(X, y, batch_size=64, epochs=100, verbose=1,validation_split=0.3)
        for x in range(pred_days):
            x_input = np.array(new_data[-7:])
            x_input = x_input.reshape((1, n_steps, n_features))
            pred = model.predict(x_input, verbose=1)
            predictions[x].append(pred[0][0])
            new_data = np.append(new_data,pred)
    avg_result = [np.mean(num_list) for num_list in predictions]
    # print("AVERAGE RESULT LSTM: " + str(avg_result)) # Printer liste av predictions
    # TODO: prøv å plotte rå data med predikerte dager i tillegg, se notebook?

    return avg_result

    

    # plt.figure(figsize=(10, 6))
    # plt.plot(history.history['mse'], label='mse')
    # plt.plot(history.history['loss'], label='loss')
    # plt.legend()
    # plt.show()
    return avg_result



    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title('model train vs validation loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'validation'], loc='upper right')
    # plt.show()

    # f = open("myfile.txt", "a")
    # f.write(str(scaled_yhat))
    # f.close()

    #plotCombined(data_visualisation['close'],[scaled_yhat])

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