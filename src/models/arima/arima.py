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

def arima_main(nr_days):
        
    today = date.today()
    d1 = today.strftime('%Y-%m-%d')
    end_date = d1
    d2 = date.today() - timedelta(days=365)
    d2 = d2.strftime('%Y-%m-%d')
    start_date = d2

    scaled_raw_data = yf.download('AAPL',
                        start=start_date,
                        end=end_date,
                        progress=False)

    scaled_raw_data.dropna()
    scaled_raw_data = scaled_raw_data.reset_index()
    #print("INDEX DATA.RESET ------ HEAD BELOW")
    #print(scaled_raw_data.head())
    scaled_raw_data['Date'] = scaled_raw_data.index
    scaled_raw_data['Date'] = pd.to_datetime(scaled_raw_data.Date, format='%Y-%m-%d')
    scaled_raw_data.index = scaled_raw_data['Date']
    scaled_raw_data = scaled_raw_data[['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
    scaled_raw_data.reset_index(drop=True, inplace=True)
    #print("DATA TAIL() ->>>>>>>", scaled_raw_data.tail())
    #print("-------HEAD BELOW-------")
    #print(scaled_raw_data.head())
    # print("--------TAIL BELOW---------")
    # print(scaled_raw_data.tail())
    scaled_raw_data = scaled_raw_data[["Date", "Close"]]
    #print("DATA TAIL: ------", scaled_raw_data.tail())


    scaled_raw_data = scaled_raw_data[['Date', 'Close']]
    #print("--------DATE & CLOSE TAIL BELOW -----------")
    #print(scaled_raw_data.head(), scaled_raw_data.tail())


    #data only containing 'Close' Column
    new_dfclose = scaled_raw_data['Close']
    #print("NEW_DFCLOSE BELOW \n\n",new_dfclose)

    # -------------PLOT FOR ORIGINAL GRAPH WITH ONLY CLOSE AND DATE
    # plt.style.use('fivethirtyeight')
    # plt.figure(figsize=(15,10))
    # plt.plot(scaled_raw_data['Date'], scaled_raw_data['Close'])
    # plt.show()


    # result = seasonal_decompose(scaled_raw_data['Close'], model='multiplicative', period=30)
    # print("RESULT FROM SEASONAL_DECOMPOSE\n", result)
    # fig = plt.figure()
    # fig = result.plot()
    # fig.set_size_inches(15,10)
    # plt.show()

    # pd.plotting.autocorrelation_plot(scaled_raw_data['Close'])
    # plt.show()


    # plot_pacf(scaled_raw_data['Close'], lags=100)
    # plt.show()

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
    #print(model_autoARIMA.summary())

    # auto_arima gives values for best model = ARIMA(0,1,0)(0,0,0)

    p, d, q = 0, 1, 0
    # inspo used SARIMAX -> to make into ARIMA some small changes must be made
    model = sm.tsa.statespace.SARIMAX(scaled_raw_data['Close'],
                                        order=(p,d,q),
                                        seasonal_order=(p,d,q,12))

    model = model.fit()

    #print(model.summary())
    # output from the predictions -> second parameter is number of days to predict
    predictions = model.predict(len(scaled_raw_data), len(scaled_raw_data) + nr_days)


    # scaled_raw_data['Close'].plot(legend=True, label='Training Data', figsize=(15,10))
    # predictions.plot(legend=True, label='Predictions')
    # plt.legend()
    # plt.show()
    return predictions._values.tolist()
