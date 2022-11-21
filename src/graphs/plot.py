import matplotlib.pyplot as plt
import numpy as np
import time as tm
import datetime as dt
from yahoo_fin import stock_info as yf


STOCK = 'AMZN'

date_now = tm.strftime('%Y-%m-%d')
yesterday  = dt.date.today() - dt.timedelta(days=2)
date_3_years_back = (yesterday - dt.timedelta(days=1104)).strftime('%Y-%m-%d')

init_df = yf.get_data(
    STOCK, 
    start_date=date_3_years_back, 
    end_date=date_now, 
    interval='1d')

init_df = init_df.drop(['open', 'high', 'low', 'adjclose', 'ticker', 'volume'], axis=1)
init_df['date'] = init_df.index

data = init_df['close']
data = np.append(data,0.005)
today = str(tm.strftime('%Y-%m-%d'))

plt.style.use(style='seaborn-v0_8')
plt.figure(figsize=(16,10))
plt.plot(init_df['close'][-200:])
plt.xlabel("days")
plt.ylabel("price")
plt.legend([f'Actual price for {STOCK}'])
plt.show()


x = []
y = []

#for line in open("src/data/data.csv", "r"):
#   lines = [i for i in line.split()]
#   x.append(int(lines[0]))
#   y.append(int(lines[1]))

#plt.title("Stock Prediction")
#plt.xlabel('Date/Days')
#plt.ylabel('Price')
#plt.yticks(y)
#plt.plot(x, y, marker = 'o', c = 'g')
  
#plt.show()
