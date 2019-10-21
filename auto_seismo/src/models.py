#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 15:35:41 2018

@author: jorgeagr
"""
import os
import numpy as np
from tensorflow.losses import huber_loss
from keras.optimizers import Adam
from keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, BatchNormalization
from keras.models import Sequential
from keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint
from aux_funcs import check_String

def load_Data(config):
    
    train_dir = check_String(config['train_dir'])
    seismos_train = check_String(config['seismos_train'])
    arrivals_train = check_String(config['arrivals_train'])
    seismos_test = check_String(config['seismos_test'])
    arrivals_test = check_String(config['arrivals_test'])
    
    data = {'train_x': np.load(train_dir + seismos_train),
            'train_y': np.load(train_dir + arrivals_train),
            'test_x': np.load(train_dir + seismos_test),
            'test_y': np.load(train_dir + arrivals_test)}
    
    if config['debug_mode']:
        for key in data:
            data[key] = data[key][:100]
    
    return data

def rossNet(seismogram_length):
    '''
    Notes
    ------------
    Ref: https://doi.org/10.1029/2017JB015251 
    '''
    model = Sequential()
    model.add(Conv1D(32, 21, activation='relu',))
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
    
    model.compile(loss=huber_loss,
                  optimizer=Adam())
    
    return model

def get_Callbacks(epochs):
    
    stopper = EarlyStopping(monitor='val_loss', min_delta=0.001, 
                            patience=epochs//2, restore_best_weights=True)
    # Include Checkpoint? CSVLogger?
    return [stopper,]

def pred_Time_Model(config):
    
    model_name = check_String(config['model_name'])
    debug_mode = config['debug_mode']
    batch_size = config['batch_size']
    epochs = config['epochs']
    model_iters = config['model_iters']
    
    if debug_mode:
        epochs=10
        model_iters=1
    
    data = load_Data(config)
    
    models = []
    models_train_means = np.zeros(model_iters)
    models_train_stds = np.zeros(model_iters)
    models_test_means = np.zeros(model_iters)
    models_test_stds = np.zeros(model_iters)
    models_test_final_loss = np.zeros(model_iters)
    
    models_train_lpe = np.zeros((model_iters, epochs))
    models_test_lpe = np.zeros((model_iters, epochs))

    for m in range(model_iters):        
        print('Training arrival prediction model', m+1)
        model = rossNet(len(data['test_x'][0]))
        
        callbacks = get_Callbacks(epochs)
        train_hist = model.fit(data['train_x'], data['train_y'],
                               validation_data=(data['test_x'], data['test_y']),
                               batch_size=batch_size,
                               epochs=epochs,
                               verbose=2,
                               callbacks=callbacks)
        
        total_epochs = len(train_hist.history['loss'])
        
        train_pred = model.predict(data['train_x'])
        test_pred = model.predict(data['test_x'])
        test_loss = model.evaluate(data['test_x'], data['test_y'],
                                   batch_size=batch_size, verbose=0)
        
        model_train_diff = np.abs(data['train_y'] - train_pred)
        model_test_diff = np.abs(data['test_y'] - test_pred)
        model_train_mean = np.mean(model_train_diff)
        model_train_std = np.std(model_train_diff)
        model_test_mean = np.mean(model_test_diff)
        model_test_std = np.std(model_test_diff)
        
        print('Train Error:', model_train_mean, '+/-', model_train_std)
        print('Test Error:', model_test_mean, '+/-', model_test_std)
        print('Test Loss:', test_loss)
        
        models.append(model)
        models_train_means[m] += model_train_mean
        models_train_stds[m] += model_train_std
        models_test_means[m] += model_test_mean
        models_test_stds[m] += model_test_std
        models_test_final_loss[m] += test_loss
        models_train_lpe[m][:total_epochs] = train_hist.history['loss']
        models_test_lpe[m][:total_epochs] = train_hist.history['val_loss']
    
    #best_model = np.argmin(models_means)
    best_model = np.argmin(models_train_means)
    print('Using best model: Model', best_model + 1)
    print('Best Model Results:')
    print('Training Avg Diff:', models_train_means[best_model])
    print('Training Avg Diff Uncertainty :', models_train_stds[best_model])
    print('Testing Avg Diff:', models_test_means[best_model])
    print('Testing Avg Diff Uncertainty:', models_test_stds[best_model])
    print('Test Loss:', models_test_final_loss[best_model])
    print('\n')
    if debug_mode:
        print('model saved in no debug')
        return
    model = models[best_model]
    model.save('./models/' + model_name + '.h5')
    np.savez('./models/etc/' + model_name + '_training_logs', loss=models_train_lpe,
             val_loss=models_test_lpe, best_model=best_model)
    
    return