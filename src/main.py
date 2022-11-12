from sklearn.preprocessing import MinMaxScaler
import time as tm
import datetime as dt
from yahoo_fin import stock_info as yf
import numpy as np
from models.lstm.lstm import lstm_main
from models.bidirectionallstm.bilstm import bi_lstm_main

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

    scaler = MinMaxScaler()
    data = get_clean_data(stock)
    data_visualisation = data.copy()
    data['close'] = scaler.fit_transform(np.expand_dims(data['close'].values, axis=1))
    raw_seq = data['close']

    #input is data - pred_days - runs
    # scaled_lstm =  lstm_main(raw_seq,1,1,True)
    # descaled_lstm = []
    # for avg in scaled_lstm:
    #     descaled_lstm.append(scaler.inverse_transform(np.array(avg).reshape(-1,1))[0][0])

    scaled_bi_lstm =  bi_lstm_main(raw_seq,2,1,False)
    descaled_bi_lstm= []
    for avg in scaled_bi_lstm:
        descaled_bi_lstm.append(scaler.inverse_transform(np.array(avg).reshape(-1,1))[0][0])

    #print(descaled_lstm)
    print(descaled_bi_lstm)
main()

