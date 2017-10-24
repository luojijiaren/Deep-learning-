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
m.add(LSTM(4,activation='sigmoid',input_shape=(None,1)))
m.add(Dense(1))
m.compile('adam',loss='mean_squared_error')

#fit
m.fit(x,y,epochs=50)

#predict
t=pd.read_csv('Google_Stock_Price_Test.csv')
real_price=t.iloc[:,1:2].values
test=real_price
test=sc.transform(test)
test=np.reshape(test,(20,1,1))
price=m.predict(test)
p=sc.inverse_transform(price)

#visulize
import matplotlib.pyplot as plt
plt.plot(real_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(p, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()



