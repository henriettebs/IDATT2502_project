from sklearn.preprocessing import MinMaxScaler,StandardScaler
import time as tm
import datetime as dt
from yahoo_fin import stock_info as yf
import numpy as np
from models.lstm.lstm import lstm_main
from models.bilstm.bilstm import bi_lstm_main
from graphs.graphs import make_graph
import matplotlib.pyplot as plt
import pandas as pd

def get_time_interval():
    date_start = (dt.date.today() - dt.timedelta(days=3)).strftime('%Y-%m-%d')
    date_end = (dt.date.today() - dt.timedelta(days=1000)).strftime('%Y-%m-%d')
    return date_start, date_end

def scale_data(data):
    scaler = StandardScaler()
    data['close'] = scaler.fit_transform(np.expand_dims(data['close'].values, axis=1))
    return data

def get_clean_data(stock):
    date_start,date_end = get_time_interval()
    init_df = yf.get_data(
    stock, 
    start_date=date_end, 
    end_date=date_start,
    interval='1d')

    init_df = init_df.drop(['open', 'high', 'low', 'adjclose', 'ticker', 'volume'], axis=1)
    init_df['date'] = init_df.index
    return init_df,date_start,date_end

def main():
    stock = 'AAPL'
    scaler = MinMaxScaler()
    data,date_start,date_end = get_clean_data(stock)
    data_visualisation = data.copy()
    data['close'] = scaler.fit_transform(np.expand_dims(data['close'].values, axis=1))
    raw_seq = data['close']

    #input is data - pred_days - runs
    scaled_lstm,lstm_history =  lstm_main(raw_seq,3,1,False)
    descaled_lstm = []
    for avg in scaled_lstm:
        descaled_lstm.append(scaler.inverse_transform(np.array(avg).reshape(-1,1))[0][0])
    
    scaled_lstm_attention,lstm_attention_history =  lstm_main(raw_seq,3,1,True)
    descaled_lstm_attention = []
    for avg in scaled_lstm_attention:
        descaled_lstm_attention.append(scaler.inverse_transform(np.array(avg).reshape(-1,1))[0][0])

    scaled_bi_lstm,bi_lstm_history =  bi_lstm_main(raw_seq,3,1,False)
    descaled_bi_lstm = []
    for avg in scaled_bi_lstm:
        descaled_bi_lstm.append(scaler.inverse_transform(np.array(avg).reshape(-1,1))[0][0])
    print("done with third \n")

    scaled_bi_lstm_attention,bi_lstm_attention_history =  bi_lstm_main(raw_seq,3,1,True)
    descaled_bi_lstm_attention = []
    for avg in scaled_bi_lstm_attention:
        descaled_bi_lstm_attention.append(scaler.inverse_transform(np.array(avg).reshape(-1,1))[0][0])
    print("done with \n")
    
    plt.figure(figsize=(10, 6))
    plt.title('LSTM')
    plt.plot(lstm_history.history['loss'], label='LSTM')
    plt.plot(lstm_attention_history.history['loss'], label='LSTM-A')
    #plt.plot(bi_lstm_history.history['loss'], label='BI-LSTM')
    #plt.plot(bi_lstm_attention_history.history['loss'], label='BI-LSTM-A')
    plt.legend()
    plt.show()


    #make_graph(data_visualisation,[[50,25,35],[150,150,150],[100,100,100]],stock)
    writeFile = open("aapl_values.txt", "w")
    writeFile.write(str(date_end) + " : " + str(date_start) + "\n")

    writeFile.write("LSTM: \n")
    writeFile.writelines(str(descaled_lstm) + "\n")

    writeFile.write("LSTM-A: \n")
    writeFile.writelines(str(descaled_lstm_attention) + "\n")

    writeFile.write("BI-LSTM: \n")
    writeFile.writelines(str(descaled_bi_lstm) + "\n")

    writeFile.write("BI-LSTM-A: \n")
    writeFile.writelines(str(descaled_bi_lstm_attention) + "\n")

    writeFile.close()

    # readFile = open("src/graphs/main.txt", "r")
    # pred = readFile.readlines()
    # print(raw_seq)
    # print(pred)

    # readFile.close()

main()

