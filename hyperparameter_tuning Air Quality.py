#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 09:36:19 2024

@author: maddy16
"""
# Regression problem to determine air quality PM2.5 from other factors like temperature, max and min temp etc.

!pip install keras-tuner
import keras
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from kerastuner.tuners import RandomSearch # help in finding the number of hidden layers to be chosen

df=pd.read_csv('Real_Combine.csv')

df.head()

X=df.iloc[:,:-1] ## independent features
y=df.iloc[:,-1] ## dependent features

#custom function- hidden layers can have 2 to 20
#hyperparameters- # of hidden layers, # of neurons in hidden layers, #learning rate
def build_model(hp):
    model = keras.Sequential()
    for i in range(hp.Int('num_layers', 2, 20)):
        model.add(layers.Dense(units=hp.Int('units_' + str(i),#Dense creates nodes/neurons inside the hidden layer
                                            min_value=32, #no. of neurons can range from 32 to 512
                                            max_value=512,
                                            step=32),
                               activation='relu'))
    model.add(layers.Dense(1, activation='linear'))#adding output layer
    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),#learning rate ranges for randomsearch
        loss='mean_absolute_error',
        metrics=['mean_absolute_error'])
    return model

tuner = RandomSearch(
    build_model,
    objective='val_mean_absolute_error',
    max_trials=5,
    executions_per_trial=3,
    directory='project',
    project_name='Air Quality Index')

tuner.search_space_summary()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

tuner.search(X_train, y_train,
             epochs=5,
             validation_data=(X_test, y_test))

tuner.results_summary()
