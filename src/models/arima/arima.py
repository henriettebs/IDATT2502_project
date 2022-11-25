import pandas as pd
import numpy as np
import yfinance as yf
import datetime
from datetime import date, timedelta
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_pacf
from pmdarima.arima import auto_arima
import statsmodels.api as sm
import math
import warnings

def arima_main(nr_days):


    today = date.fromisoformat('2022-11-20')
    end_date = today
    d2 = date.fromisoformat('2020-06-06')
    start_date = d2

    scaled_raw_data = yf.download('MSFT',
                        start=start_date,
                        end=end_date,
                        progress=False)

    scaled_raw_data.dropna()
    scaled_raw_data = scaled_raw_data.reset_index()
    print("Date column exists and indexes on each element\n",scaled_raw_data.head())
    scaled_raw_data = scaled_raw_data[["Date", "Close"]]
    new_dfclose = scaled_raw_data['Close']

    # -------------PLOT FOR ORIGINAL GRAPH WITH ONLY CLOSE AND DATE
    plt.style.use('fivethirtyeight')
    plt.figure(figsize=(15,10))
    plt.plot(scaled_raw_data['Date'], scaled_raw_data['Close'])
    

    # plot is just blank
    result = seasonal_decompose(scaled_raw_data['Close'], model='multiplicative', period=30)
    print("RESULT FROM SEASONAL_DECOMPOSE\n", result)
    fig = plt.figure()
    fig = result.plot()
    fig.set_size_inches(15,10)

    pd.plotting.autocorrelation_plot(scaled_raw_data['Close'])

    plot_pacf(scaled_raw_data['Close'], lags=100)

    #Setup auto_arima for p, d, q, values
    df_log = np.log(scaled_raw_data['Close'])
    #print("DF_LOG: \n", df_log)
    moving_avg = df_log.rolling(12).mean()
    std_dev = df_log.rolling(12).std()

    train_data, test_data = df_log[3:int(len(df_log)*0.9)], df_log[int(len(df_log)*0.9):]


    model_autoARIMA = auto_arima(train_data, start_p=0, start_q=0,
                                test='adf',
                                max_p=7, max_q=7,
                                m=1,
                                d=None,
                                seasonal=True,
                                start_P=0,
                                D=None,
                                trace=True,
                                error_action='ignore',
                                suppress_warnings=True,
                                stepwise=True)
    print(model_autoARIMA.summary())

    # auto_arima gives values for best model = ARIMA(0,1,0)(0,0,0)

    p, d, q = 0, 1, 0
    # inspo used SARIMAX -> to make into ARIMA some small changes must be made
    model = sm.tsa.statespace.SARIMAX(scaled_raw_data['Close'],
                                        order=(p,d,q),
                                        seasonal_order=(p,d,q,12))

    model = model.fit()

    print(model.summary())
    # output from the predictions -> second parameter is number of days to predict
    predictions = model.predict(len(scaled_raw_data), len(scaled_raw_data) + nr_days)
    pred_result = predictions._values.tolist()


    scaled_raw_data['Close'].plot(legend=True, label='Training Data', figsize=(15,10))
    predictions.plot(legend=True, label='Predictions')
    plt.legend()
    plt.show()
    return pred_result
