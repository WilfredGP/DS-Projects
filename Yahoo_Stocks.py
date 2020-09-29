import pandas as pd
import pandas_datareader as web
from datetime import datetime

df_tsla = web.DataReader('TSLA', data_source='yahoo', start='2000-01-01', end=datetime.date(datetime.now()))
df_tsla.reset_index(inplace=True)

df_aapl = web.DataReader('AAPL', data_source='yahoo', start='2000-01-01', end=datetime.date(datetime.now()))
df_aapl.reset_index(inplace=True)

df_spy = web.DataReader('SPY', data_source='yahoo', start='2000-01-01', end=datetime.date(datetime.now()))
df_spy.reset_index(inplace=True)

df_msft = web.DataReader('MSFT', data_source='yahoo', start='2000-01-01', end=datetime.date(datetime.now()))
df_msft.reset_index(inplace=True)

df_nvda = web.DataReader('NVDA', data_source='yahoo', start='2000-01-01', end=datetime.date(datetime.now()))
df_nvda.reset_index(inplace=True)

df_qqq = web.DataReader('QQQ', data_source='yahoo', start='2000-01-01', end=datetime.date(datetime.now()))
df_qqq.reset_index(inplace=True)

