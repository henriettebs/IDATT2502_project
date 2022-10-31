import numpy as np
import time as tm
import datetime as dt
import tensorflow as tf
from yahoo_fin import stock_info as yf
from sklearn.preprocessing import MinMaxScaler
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import matplotlib.pyplot as plt


N_STEPS = 7


class Lstm:
    def __init__(self,stock,data):
        self.stock =  stock
        self.df = data

    def Model(x_train, y_train):
        model = Sequential()
        model.add(LSTM(60, return_sequences=True, input_shape=(N_STEPS, len(['close']))))
        model.add(Dropout(0.5))
        model.add(LSTM(120, return_sequences=False))
        model.add(Dropout(0.5))
        model.add(Dense(20))
        model.add(Dense(1))
        BATCH_SIZE = 8
        EPOCHS = 100
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(x_train, y_train,
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    verbose=1)
        model.summary()
        return model

    def Train(self,days):
        df = df.copy()
        df['future'] = df['close'].shift(-days)
        last_sequence = np.array(df[['close']].tail(days))
        df.dropna(inplace=True)
        sequence_data = []
        sequences = deque(maxlen=N_STEPS)

        for entry, target in zip(df[['close'] + ['date']].values, df['future'].values):
            sequences.append(entry)
            if len(sequences) == N_STEPS:
                sequence_data.append([np.array(sequences), target])

        last_sequence = list([s[:len(['close'])] for s in sequences]) + list(last_sequence)
        last_sequence = np.array(last_sequence).astype(np.float32)

        # construct the X's and Y's
        X, Y = [], []
        for seq, target in sequence_data:
            X.append(seq)
            Y.append(target)

        # convert to numpy arrays
        X = np.array(X)
        Y = np.array(Y)

        return df, last_sequence, X, Y


STOCK = "AAPL"

date_now = tm.strftime('%Y-%m-%d')
date_3_years_back = ((dt.date.today() - dt.timedelta(days=1)) - dt.timedelta(days=1104)).strftime('%Y-%m-%d')

df = yf.get_data(
    STOCK, 
    start_date=date_3_years_back, 
    end_date=date_now, 
    interval='1d')

df = df.drop(['open', 'high', 'low', 'adjclose', 'ticker', 'volume'], axis=1)
df['date'] = df.index

plt.style.use(style='ggplot')
plt.figure(figsize=(16,10))
plt.plot(df['close'][-200:])
plt.xlabel("days")
plt.ylabel("price")
plt.legend([f'Actual price for {STOCK}'])
plt.show()

scaler = MinMaxScaler()
df['close'] = scaler.fit_transform(np.expand_dims(df['close'].values, axis=1))

lstm = Lstm("AAPL",df)


predictions = []
days = [1]

for step in days:
  df, last_sequence, x_train, y_train = lstm.Train(step)
  x_train = x_train[:, :, :len(['close'])].astype(np.float32)

  model = lstm.Model(x_train,y_train)

  last_sequence = last_sequence[-N_STEPS:]
  last_sequence = np.expand_dims(last_sequence, axis=0)
  prediction = model.predict(last_sequence)
  predicted_price = scaler.inverse_transform(prediction)[0][0]

  predictions.append(round(float(predicted_price), 2))

  print(predictions)