#Scale the data
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd

def compute_stock_features(X):
    x = []
    y = []
    for i in range(60,len(X)):
        x.append(X[i-60:i,0])
        y.append(X[i,0])
    x, y = np.array(x), np.array(y)
    x = np.reshape(x, (x.shape[0],x.shape[1],1))
    return x, y


