# Recurrent Neural Network

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the training set
training_set = pd.read_csv('Google_Stock_Price_Train.csv')
training = training_set.iloc[:,1:2].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
training_set = sc.fit_transform(training)

# Getting the inputs and the ouputs
X_train=[]
y_train=[]
for i in range(20,1258):
    X_train.append(training_set[i-20:i,0])
    y_train.append(training_set[i,0])
X_train,y_train=np.array(X_train),np.array(y_train)

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))  #add a number of feature 

# Part 2 - Building the RNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

# Initialising the RNN
regressor = Sequential()

# Adding the input layer and the LSTM layer
regressor.add(LSTM(units = 3, input_shape = (None, 1),return_sequences=True))
regressor.add(LSTM(units = 3, return_sequences=True))
regressor.add(LSTM(units = 3, return_sequences=True))
regressor.add(LSTM(units = 3))
# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'rmsprop', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, batch_size = 32, epochs = 100)

# Part 3 - Making the predictions and visualising the results

# Getting the real stock price of 2017
test_set = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = test_set.iloc[:,1:2].values
r=np.concatenate((training[0:1258],real_stock_price),axis=0)

# Getting the predicted stock price of 2017
stock = r
stock = sc.fit_transform(stock)
inputs=[]
for i in range(1258,1278):
    inputs.append(stock[i-20:i,0])
inputs=np.array(inputs)

inputs = np.reshape(inputs, (inputs.shape[0], inputs.shape[1], 1))
predicted_stock_price = regressor.predict(inputs)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualising the results
plt.plot(r[1258:], color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()