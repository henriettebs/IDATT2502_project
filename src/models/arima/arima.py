 
#Kilde brukt for INSPO til Arima "SARIMAX"
# https://www.kaggle.com/code/sainischala/stock-prices-predictor-using-arima


import pandas as pd
import numpy as np
import yfinance as yf
import datetime
from datetime import date, timedelta
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_pacf
from pmdarima.arima import auto_arima
import statsmodels.api as sm
import warnings

today = date.today()
d1 = today.strftime('%Y-%m-%d')
end_date = d1
d2 = date.today() - timedelta(days=365)
d2 = d2.strftime('%Y-%m-%d')
start_date = d2

data = yf.download('AAPL',
                    start=start_date,
                    end=end_date,
                    progress=False)
#data.dropna()
#new_data = data.reset_index()
#print("INDEX DATA.RESET ------ HEAD BELOW")
#print(new_data.head())
data['Date'] = data.index
#new_data['Date'] = pd.to_datetime(new_data.Date, format='%Y-%m-%d')
#new_data.index = new_data['Date']
data = data[['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
data.reset_index(drop=True, inplace=True)
print("DATA TAIL() ->>>>>>>", data.tail())
#print("-------HEAD BELOW-------")
#print(new_data.head())
#print("--------TAIL BELOW---------")
#print(new_data.tail())
data = data[["Date", "Close"]]
print("DATA TAIL: ------", data.tail())


#new_data = new_data[['Date', 'Close']]
#print("--------DATE & CLOSE TAIL BELOW -----------")
#print(new_data.head(), new_data.tail())


#data only containing 'Close' Column
#new_dfclose = new_data['Close']
#print("NEW_DFCLOSE BELOW \n\n",new_dfclose)

# -------------PLOT FOR ORIGINAL GRAPH WITH ONLY CLOSE AND DATE
plt.style.use('fivethirtyeight')
plt.figure(figsize=(15,10))
plt.plot(data['Date'], data['Close'])
plt.show()


result = seasonal_decompose(data['Close'], model='multiplicative', period=30)
print("RESULT FROM SEASONAL_DECOMPOSE\n", result)
#fig = plt.figure()
#fig = result.plot()
#fig.set_size_inches(15,10)
#plt.show()

pd.plotting.autocorrelation_plot(data['Close'])
#plt.show()


plot_pacf(data['Close'], lags=100)
#plt.show()

#Setup auto_arima for p, d, q, values
#df_log = np.log(new_dfclose)
#print("DF_LOG: \n", df_log)
#moving_avg = df_log.rolling(12).mean()
#std_dev = df_log.rolling(12).std()

#train_data, test_data = df_log[3:int(len(df_log)*0.9)], df_log[int(len(df_log)*0.9):]


# model_autoARIMA = auto_arima(train_data, start_p=0, start_q=0,
#                             test='adf',
#                             max_p=7, max_q=7,
#                             m=1,
#                             d=None,
#                             seasonal=True,
#                             start_P=0,
#                             D=None,
#                             trace=True,
#                             error_action='ignore',
#                             suppress_warnings=True,
#                             stepwise=True)
# print(model_autoARIMA.summary())

# auto_arima gives values for best model = ARIMA(0,1,0)(0,0,0)

p, d, q = 0, 1, 0
# inspo used SARIMAX -> to make into ARIMA some small changes must be made
model = sm.tsa.statespace.SARIMAX(data['Close'],
                                    order=(p,d,q),
                                    seasonal_order=(p,d,q,12))

model = model.fit()
print(model.summary())


# fc, se, conf = model.forecast(3, alpha=0.05)
# sconf = str(conf)
# fc_series = pd.Series(fc, index=test_data.index)
# #lower_series
# #upper_series
# plt.figure(figsize=(12,5), dpi=100)
# plt.plot(train_data, label="Training")
# plt.plot(test_data, color="blue", label="Actual stock price")
# plt.plot(fc_series, color="red", label="Predicted stock price")
# #plt.fill_between()
# plt.title("AAPL Stock Price Prediction")
# plt.xlabel("Time")
# plt.ylabel("Close Stock Price")
# plt.legend(loc="upper left")
# plt.show()

predictions = model.predict(len(data), len(data) + 10)
print(predictions)

data['Close'].plot(legend=True, label='Training Data', figsize=(15,10))
predictions.plot(legend=True, label='Predictions')
plt.legend()
plt.show()
