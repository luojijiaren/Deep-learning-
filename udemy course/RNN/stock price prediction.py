# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 12:55:56 2017

@author: fzhan
"""

import keras
import pandas as pd

data=pd.read_csv('Google_Stock_Price_Train.csv')
data=data.iloc[:,1:2].values

#scale
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
training=sc.fit_transform(data)

x=training[0:1257]
y=training[1:1258]

#reshape
import numpy as np
x=np.reshape(x,(1257,1,1))

#rnn
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense

m=Sequential()
m.add(LSTM(4,activation='sigmoid',))

