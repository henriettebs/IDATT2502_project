import numpy as np
import pandas as pd

import os
import matplotlib.pyplot as plt
import datetime as dt
from yahoo_fin import stock_info as yf
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM,Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from layers.attention import attention


def get_time_interval(s,e):
    date_start = (dt.date.today() - dt.timedelta(days=s)).strftime('%Y-%m-%d')
    date_end = (dt.date.today() - dt.timedelta(days=e)).strftime('%Y-%m-%d')
    return date_start, date_end

def get_clean_data(stock,s,e):
    date_start,date_end = get_time_interval(s,e)
    init_df = yf.get_data(
    stock, 
    start_date=date_end, 
    end_date=date_start,
    interval='1d')

    init_df = init_df.drop(['open', 'high', 'low', 'adjclose', 'ticker', 'volume'], axis=1)
    init_df['date'] = init_df.index
    return init_df,date_start,date_end

def LSTM_model(x_train):
    
    model = Sequential()
    
    model.add(LSTM(units = 50, return_sequences = False, input_shape = (x_train.shape[1],1)))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    
    return model


def main():
    stock = 'AAPL'
    scaler = MinMaxScaler()
    data,date_start,date_end = get_clean_data(stock,100,1000)
    pred_data,pred_date_start,pred_date_end = get_clean_data(stock,0,200)
    scaled_data = scaler.fit_transform(data['close'].values.reshape(-1,1))
    print(scaled_data)

    prediction_days = 20

    x_train = []
    y_train = []

    for x in range(prediction_days, len(scaled_data)):
        x_train.append(scaled_data[x - prediction_days:x, 0])
        y_train.append(scaled_data[x, 0])
        
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    model = LSTM_model(x_train)
    model.summary()
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='mean_squared_error')
    history = model.fit(x_train, y_train, epochs=70, batch_size = 64,validation_split= 0.2)


    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='loss')
    plt.legend()
    plt.show()



    actual_prices = pred_data['close'].values

    total_dataset = pd.concat((data['close'], pred_data['close']), axis=0)
    print(len(total_dataset),len(pred_data),prediction_days)
    model_inputs = total_dataset[len(total_dataset) - len(pred_data) - prediction_days:].values
    print(len(model_inputs))
    model_inputs = model_inputs.reshape(-1,1)
    model_inputs = scaler.transform(model_inputs)
    # print(model_inputs)
    # x_test = []
    # for x in range(prediction_days, len(model_inputs)):
    #     x_test.append(model_inputs[x-prediction_days:x, 0])
    
    # print(x_test)
    # print(len(x_test))

    # x_test = np.array(x_test)
    # x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1] ,1))
    # print(x_test)
    # print(x_test.shape)
    # predicted_prices = model.predict(x_test)
    # predicted_prices = scaler.inverse_transform(predicted_prices)

    # plt.plot(actual_prices, color='black', label=f"Actual {stock} price")
    # plt.plot(predicted_prices, color= 'green', label=f"predicted {stock} price")
    # plt.title(f"{stock} share price")
    # plt.xlabel("time")
    # plt.ylabel(f"{stock} share price")
    # plt.legend()
    # plt.show()


    real_data = [model_inputs[len(model_inputs)+1 - prediction_days:len(model_inputs)+1,0]]
    real_data = np.array(real_data)
    real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))
    

    prediction = model.predict(real_data)
    prediction = scaler.inverse_transform(prediction)
    print(f"prediction: {prediction}")

main()