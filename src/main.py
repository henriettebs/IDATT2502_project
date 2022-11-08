from sklearn.preprocessing import MinMaxScaler
import time as tm
import datetime as dt
from yahoo_fin import stock_info as yf
import numpy as np
from models.Lstm.lstm import lstm_main
from models.LstmAttention.lstmAttention import lstmAttentionMain
def getTimeInterval():
    date_now = tm.strftime('%Y-%m-%d')
    date_3_years_back = ((dt.date.today() - dt.timedelta(days=2)) - dt.timedelta(days=1200)).strftime('%Y-%m-%d')
    return date_now, date_3_years_back

def scaleData(data):
    scaler = MinMaxScaler()
    data['close'] = scaler.fit_transform(np.expand_dims(data['close'].values, axis=1))
    return data

def getCleanData(stock):
    date_now,date_3_years_back = getTimeInterval()
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
    data = getCleanData(stock)
    data_visualisation = data.copy()
    data['close'] = scaler.fit_transform(np.expand_dims(data['close'].values, axis=1))
    raw_seq = data['close']

    #input is data - pred_days - runs
    scaled_lstm =  lstm_main(raw_seq,2,1)
    descaled_lstm = []
    for avg in scaled_lstm:
        descaled_lstm.append(scaler.inverse_transform(np.array(avg).reshape(-1,1))[0][0])

    scaled_lstm_attention =  lstmAttentionMain(raw_seq,2,1)
    descaled_lstm_attention = []
    for avg in scaled_lstm_attention:
        descaled_lstm_attention.append(scaler.inverse_transform(np.array(avg).reshape(-1,1))[0][0])

    print(descaled_lstm)
    print(descaled_lstm_attention)
main()

