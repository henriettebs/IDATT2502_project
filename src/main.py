from sklearn.preprocessing import MinMaxScaler
import time as tm
import datetime as dt
from yahoo_fin import stock_info as yf
import numpy as np
from models.lstm.lstm import lstm_main
from models.bilstm.bilstm import bi_lstm_main
import matplotlib.pyplot as plt
def get_time_interval():
    date_now = tm.strftime('%Y-%m-%d')
    date_3_years_back = ((dt.date.today() - dt.timedelta(days=2)) - dt.timedelta(days=1200)).strftime('%Y-%m-%d')
    return date_now, date_3_years_back

def scale_data(data):
    scaler = MinMaxScaler()
    data['close'] = scaler.fit_transform(np.expand_dims(data['close'].values, axis=1))
    return data

def get_clean_data(stock):
    date_now,date_3_years_back = get_time_interval()
    init_df = yf.get_data(
    stock, 
    start_date=date_3_years_back, 
    end_date=date_now,
    interval='1d')

    init_df = init_df.drop(['open', 'high', 'low', 'adjclose', 'ticker', 'volume'], axis=1)
    init_df['date'] = init_df.index
    return init_df

def main():
    stock = 'AMZN'

    dates = []
    for x in range(100):
        dates.append((dt.date.today() - dt.timedelta(days=x)).strftime('%Y-%m-%d'))

    scaler = MinMaxScaler()
    data = get_clean_data(stock)
    data_visualisation = data.copy()
    #data['close'] = scaler.fit_transform(np.expand_dims(data['close'].values, axis=1))
    #raw_seq = data['close']
    temp = data.copy()
    temp2 = data.copy()

    date_now = dt.date.today()
    date_tomorrow = dt.date.today() + dt.timedelta(days=1)
    date_after_tomorrow = dt.date.today() + dt.timedelta(days=2)

    temp.loc[date_now] = [120, f'{date_now}']
    temp.loc[date_tomorrow] = [110, f'{date_tomorrow}']
    temp.loc[date_after_tomorrow] = [90, f'{date_after_tomorrow}']

    temp2.loc[date_now] = [90, f'{date_now}']
    temp2.loc[date_tomorrow] = [70, f'{date_tomorrow}']
    temp2.loc[date_after_tomorrow] = [60, f'{date_after_tomorrow}']

    plt.style.use(style='ggplot')
    plt.figure(figsize=(16,10))
    plt.plot(temp['close'][-150:].head(147))
    plt.plot(temp['close'][-150:].tail(4))
    plt.plot(temp2['close'][-150:].tail(4))
    plt.xlabel('days')
    plt.ylabel('price')
    plt.legend([f'Actual price for {stock}', 
                f'Predicted price for future 3 days'])
    plt.show()

    #input is data - pred_days - runs
    # scaled_lstm =  lstm_main(raw_seq,1,1,True)
    # descaled_lstm = []
    # for avg in scaled_lstm:
    #     descaled_lstm.append(scaler.inverse_transform(np.array(avg).reshape(-1,1))[0][0])

    # scaled_bi_lstm =  bi_lstm_main(raw_seq,2,1,False)
    # descaled_bi_lstm= []
    # for avg in scaled_bi_lstm:
    #     descaled_bi_lstm.append(scaler.inverse_transform(np.array(avg).reshape(-1,1))[0][0])

    #print(descaled_lstm)
    #print(descaled_bi_lstm)
main()

