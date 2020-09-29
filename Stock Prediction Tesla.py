#!/usr/bin/env python
# coding: utf-8

# # Stock Prediction - LSTM 

# ## Import Libraries

# In[1]:


#Import the libraries
import math
import pandas_datareader as web
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import sklearn
from sklearn.preprocessing import MinMaxScaler
import os


plt.style.use('fivethirtyeight')
os.chdir(u'C:/Users/wgutierrezp/Documents/GitHub/DS-Projects/')
STOCK_NAME = 'TSLA'
# ## Ingest Data 

# In[13]:


#Get the stock quote 
df = web.DataReader(STOCK_NAME, data_source='yahoo', start='2000-01-01', end=datetime.date(datetime.now())) 
 
# ## Data Preprocessing & Wrangling

#Select closing data
df_close = df[['Close']]


#Scale the data
scaler = sklearn.preprocessing.MinMaxScaler()
X = scaler.fit_transform(df_close)

##
model = keras.models.load_model('models/LSTM_60D')


from utils.preprocessing import compute_stock_features
x, y = compute_stock_features(X)


# In[28]:

p_list = [np.zeros(x.shape[1]),scaler.inverse_transform(model.predict(x))]
df_close['Predictions'] = np.array([i for l in p_list for i in l])
df_close.reset_index(inplace=True)
df_close['Stock'] = STOCK_NAME


# %%
#Compute next 60 days
for i in range(60):
    y_pred = model.predict(x[-1,:])
    