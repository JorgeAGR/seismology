#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 15:35:41 2018

@author: jorgeagr
"""
import os
import numpy as np
import tensorflow as tf
import keras
from keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, BatchNormalization
from keras.models import Sequential

datadir = '.'

seismograms = np.load('seismograms.npy')
arrivals = np.load('arrivals.npy')

test_sample = 10000
cutoff = len(seismograms) - test_sample

train_x = seismograms[:cutoff]
train_y = arrivals[:cutoff]

test_x = seismograms[cutoff:]
test_y = arrivals[cutoff:]

models = []
models_means = []
models_stds = []
for m in range(10):
    print('Training arrival prediction model 1...')
    model = Sequential()
    model.add(Conv1D(32, kernel_size=21, strides=1,
                     activation='relu',
                     input_shape=(len(seismograms[0]), 1)))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    
    model.add(Conv1D(64, 15, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    
    model.add(Conv1D(128, 11, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(1, activation='linear'))
    
    model.compile(loss='mse',
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])
    
    model.fit(train_x, train_y,
              batch_size=128,
              epochs=20,
              verbose=0,)
    pred = model.predict(seismograms)
    #print(model.evaluate(test_x, test_y))
    
    model_mean = np.mean(np.abs(arrivals - pred))
    model_std = np.std(np.abs(arrivals - pred))
    
    #print('Avg Diff:', model_mean)
    #print('Avg Diff Error:', model_std)
    
    models.append(model)
    models_means.append(model_mean)
    models_stds.append(model_std)

best_model = np.argsort(models_stds)[0]
print('Using best model...')
model = models[best_model]
print('Best Model Avg Diff:', models_means[best_model])
print('Best Model Avg Diff Error:', models_stds[best_model])
