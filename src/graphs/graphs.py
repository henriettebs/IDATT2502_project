import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt

def make_graph(data,predictions, stock):
    methods = len(predictions)
    days = len(predictions[0])
    plt.style.use(style='ggplot')
    plt.figure(figsize=(16,10))
    plt.plot(data['close'][-30:])
    for x in range(methods):
        temp = data.copy()
        for y in range(days):
            date = (dt.date.today() - dt.timedelta(days=4)) + dt.timedelta(days=y)
            temp.loc[date] = [predictions[x][y], f'{date}']
        plt.plot(temp['close'].tail(4))
    plt.xlabel('days')
    plt.ylabel('price')
    plt.legend([f'Actual pre price for {stock}', 
                f'Predicted price, LSTM',
                f'Predicted price, LSTM with attention',
                f'Predicted price, BI-LSTM',
                f'Predicted price, BI-LSTM with attention',
                f'Actual after price for {stock}'])
    plt.show()