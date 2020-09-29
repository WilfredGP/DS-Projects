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
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import sklearn
from sklearn.preprocessing import MinMaxScaler
# from keras.models import Sequential
# from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')


# ## Ingest Data 

# In[13]:


#Get the stock quote 
df = web.DataReader('TSLA', data_source='yahoo', start='2000-01-01', end=datetime.date(datetime.now())) 
#Show the data 


# ## Exploratory Data Analysis

# In[14]:




# In[16]:




# ## Data Preprocessing & Wrangling

# In[17]:


#Select closing data
df_close = df[['Close']]


# In[18]:


#Scale the data
scaler = sklearn.preprocessing.MinMaxScaler()
X = scaler.fit_transform(df_close)
X.shape

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


# ## Model Training

# In[19]:


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


# ## Model Evaluation

# In[21]:


#Test data set
test_data = X[training_data_len - 60: , : ]
#Create the x_test and y_test data sets
x_test = []
y_test = scaler.inverse_transform(X[training_data_len:, :]) #Get all of the rows from index 1603 to the rest and all of the columns (in this case it's only column 'Close'), so 2003 - 1603 = 400 rows of data
for i in range(60,len(test_data)):
    x_test.append(test_data[i-60:i,0])
    
x_test = np.array(x_test)
print(x_test.shape)
print(y_test.shape)


# In[22]:


#Reshape the data into the shape accepted by the LSTM
x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))


# In[23]:


#Getting the models predicted price values
predictions = model.predict(x_test) 
predictions = scaler.inverse_transform(predictions)#Undo scaling


from utils.preprocessing import compute_stock_features

x, y = compute_stock_features(X)


# In[28]:

p_list = [np.zeros(x.shape[1]),scaler.inverse_transform(model.predict(x))]
df_close['Predictions'] = np.array([i for l in p_list for i in l])
df_close.reset_index(inplace=True)



