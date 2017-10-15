# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 08:47:03 2017

@author: fzhan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#read data
dataset=pd.read_csv('Churn_Modelling.csv')
X=dataset.iloc[:,3:13].values
y=dataset.iloc[:,13].values

#dummy 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
x1=LabelEncoder()
X[:,1]=x1.fit_transform(X[:,1])
x2=LabelEncoder()
X[:,2]=x2.fit_transform(X[:,2])
one=OneHotEncoder(categorical_features=[1])
X=one.fit_transform(X).toarray()
X=X[:,1:]

#split data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

#scale data
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

#keras ANN model
import tensorflow as tf
import keras

from keras.models import Sequential
from keras.layers import Dense

classifier=Sequential()
classifier.add(Dense(units=6,activation='relu',kernel_initializer='uniform',input_dim=11))
classifier.add(Dense(units=6,activation='relu',kernel_initializer='uniform'))
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
classifier.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])

#fit and predict
classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)
y_pred=(y_pred>0.5)

new_pred=classifier.predict(sc.transform(np.array([[0.0,0,600,1,40,3,60000,2,1,1,50000]])))
new_pred=(new_pred>0.5)

#get confusion_matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from keras.layers import Dropout
def build_classifier():
    classifier=Sequential()
    classifier.add(Dense(units=6,activation='relu',kernel_initializer='uniform',input_dim=11))
    classifier.add(Dropout(p=0.1)
    classifier.add(Dense(units=6,activation='relu',kernel_initializer='uniform'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])
    return classifier
cl=KerasClassifier(build_fn=build_classifier,batch_size=10,nb_epoch=100)
parameters={'batch_size':[25,32],
            'nb_epoch':[100,500],
            'optimizer':['adam','rmsprop']}
grid=GridSearchCV(estimator=cl,param_grid=parameters,scoring='accuracy',cv=10)
grid=grid.fit(X_train,y_train)
best_parameters=grid.best_params_
best_accuracy=grid.best_score_

#accuracies=cross_val_score(estimator=cl,X=X_train,y=y_train,cv=10,n_jobs=-1)
#acc=accuracies.mean()
