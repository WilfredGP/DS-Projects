#!/usr/bin/env python
# coding: utf-8

# # Stock Prediction - LSTM 

# ## Import Libraries

# In[3]:


#Import the libraries
import math
import pandas_datareader as web
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import sklearn
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')


# ## Ingest Data 

# In[4]:


#Get the stock quote 
df = web.DataReader('AAPL', data_source='yahoo', start='2012-01-01', end='2019-12-17') 


#Select closing data
df_close = df[['Close']]


#Scale the data
scaler = sklearn.preprocessing.MinMaxScaler()
X = scaler.fit_transform(df_close)

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
training_data_len = math.ceil( len(df_close) *.8)
train_data = X[0:training_data_len  , : ]
X_train,X_test = train_test_split(X, test_size=0.2 ,shuffle=False)

x_train=[]
y_train = []
for i in range(60,len(X_train)):
    x_train.append(X_train[i-60:i,0])
    y_train.append(X_train[i,0])

x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

#Train Model

#Build the LSTM network model
model = keras.Sequential()
model.add(keras.layers.LSTM(units=50, return_sequences=True,input_shape=(x_train.shape[1],1)))
model.add(keras.layers.LSTM(units=50, return_sequences=False))
model.add(keras.layers.Dense(units=25))
model.add(keras.layers.Dense(units=1))

#Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

#Train the model
model.fit(x_train, y_train, batch_size=1, epochs=1)


# # Generate Output Dataframe

from utils.preprocessing import compute_stock_features

x, y = compute_stock_features(X)


# In[123]:

p_list = [np.zeros(x.shape[1]),scaler.inverse_transform(model.predict(x))]
df_close['Predictions'] = np.array([i for l in p_list for i in l])
df_close.reset_index(inplace=True)





