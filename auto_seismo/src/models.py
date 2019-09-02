#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 15:35:41 2018

@author: jorgeagr

Based on the network established by Ross et al, 2018 from
the Seismological Laboratory in Caltech

References:
Ross, Meier and Hauksson
P-wave arrival picking and first-motion polarity determination with deep learning
J. Geophys. Res.-Solid Earth
2018
"""
import os
import numpy as np
import tensorflow as tf
import keras
from keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, BatchNormalization
from keras.models import Sequential
from aux_funcs import check_string

def pred_Time_Model(train_dir, seismos_train, arrivals_train, batch_size, epochs, model_iters, debug_mode=False):
    train_dir = check_string(train_dir)
    seismos_train = check_string(seismos_train)
    arrivals_train = check_string(arrivals_train)
    
    seismograms = np.load(train_dir + seismos_train)
    arrivals = np.load(train_dir + arrivals_train)
    
    test_percent = 0.15
    
    models = []
    models_means = []
    models_stds = []
    shuffled_indeces = []
    for m in range(model_iters):
        
        if debug_mode:
            test_sample = 290000 # debug line
            
        rand_index = np.arange(0, len(seismograms), 1)
        np.random.shuffle(rand_index)
        
        seismograms = seismograms[rand_index]
        arrivals = arrivals[rand_index]
        
        cutoff = int(len(seismograms) - len(seismograms) * test_percent)
        train_x = seismograms[:cutoff]
        train_y = arrivals[:cutoff]
        
        if test_percent != 0:
            test_x = seismograms[cutoff:]
            test_y = arrivals[cutoff:]
        
        print('Training arrival prediction model', m+1)
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
        
        model.compile(loss=tf.losses.huber_loss,
                      optimizer=keras.optimizers.Adam())
        
        model.fit(train_x, train_y,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,)
        if debug_mode:
            pred = model.predict(seismograms[:100])
            arrivals = arrivals[:100]
        else:
            train_pred = model.predict(train_x)
            test_pred = model.predict(test_x)
        
        model_train_mean = np.mean(np.abs(train_y - train_pred))
        model_train_std = np.std(np.abs(train_y - train_pred))
        model_test_mean = np.mean(np.abs(test_y - test_pred))
        model_test_std = np.std(np.abs(test_y - test_pred))
        
        print('Train Error:', model_train_mean, '+/-', model_train_std)
        print('Test Error:', model_test_mean, '+/-', model_test_std)
        
        models.append(model)
        models_means.append(model_train_mean)
        models_stds.append(model_train_std)
        shuffled_indeces.append(rand_index)
        
        model.save('./models/pred_model_' + str(m) + '.h5')
    
    #best_model = np.argsort(models_means)[-1]
    best_model = np.argmin(models_means)
    print('Using best model: Model', best_model)
    model = models[best_model]
    np.save('./models/etc/model_train_test_index.npy', shuffled_indeces[best_model])
    print('Best Model Avg Diff:', models_means[best_model])
    print('Best Model Avg Diff Error:', models_stds[best_model])
    
    return model