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
from tensorflow.losses import huber_loss
import keras.backend as tf
from keras.optimizers import Adam
from keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, BatchNormalization
from keras.models import Sequential
from aux_funcs import check_String

def abs_Error(y_true, y_pred):
    return tf.mean(tf.abs(y_true - y_pred))

def sort_Data(data_X, data_y, test_percent=0.15, debug_mode=False):
            
    rand_index = np.arange(0, len(data_X), 1)
    np.random.shuffle(rand_index)
    
    cutoff = int(len(data_X) - len(data_X) * test_percent)
    train_ind = rand_index[:cutoff]
    test_ind = rand_index[cutoff:-50]
    blind_ind = rand_index[-50:]
    
    train_x = data_X[train_ind]
    train_y = data_y[train_ind]
    
    if test_percent != 0:
        test_x = data_X[test_ind]
        test_y = data_y[test_ind]
    
    if debug_mode:
        train_x = train_x[:1000]
        train_y = train_y[:1000]
        test_x = test_x[:100]
        test_y = test_y[:100]
    
    data = {'train_x': train_x, 'train_y': train_y,
            'test_x': test_x, 'test_y': test_y,
            'train_index': train_ind, 'test_index': test_ind,
            'blind_index': blind_ind}
    
    return data
    

def pred_Time_Model(model_name, train_dir, seismos_train, arrivals_train, 
                    batch_size, epochs, model_iters, debug_mode=False):
    train_dir = check_String(train_dir)
    seismos_train = check_String(seismos_train)
    arrivals_train = check_String(arrivals_train)
    
    seismograms = np.load(train_dir + seismos_train)
    arrivals = np.load(train_dir + arrivals_train)
    
    if debug_mode:
        epochs=1
        model_iters=1
    
    models = []
    models_train_means = []
    models_train_stds = []
    models_test_means = []
    models_test_stds = []
    train_indices = []
    test_indices = []
    blind_indices = []
    for m in range(model_iters):
        
        data = sort_Data(seismograms, arrivals, debug_mode=debug_mode)
        
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
        
        model.compile(loss=huber_loss,
                      optimizer=Adam(),
                      metrics=[abs_Error])
        
        model.fit(data['train_x'], data['train_y'],
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,)
        
        train_pred = model.predict(data['train_x'])
        test_pred = model.predict(data['test_x'])
        
        model_train_diff = np.abs(data['train_y'] - train_pred)
        model_test_diff = np.abs(data['test_y'] - test_pred)
        model_train_mean = np.mean(model_train_diff)
        model_train_std = np.std(model_train_diff)
        model_test_mean = np.mean(model_test_diff)
        model_test_std = np.std(model_test_diff)
        
        print('Train Error:', model_train_mean, '+/-', model_train_std)
        print('Test Error:', model_test_mean, '+/-', model_test_std)
        
        models.append(model)
        models_train_means.append(model_train_mean)
        models_train_stds.append(model_train_std)
        models_test_means.append(model_test_mean)
        models_test_stds.append(model_test_std)
        train_indices.append(data['train_index'])
        test_indices.append(data['test_index'])
        blind_indices.append(data['blind_index'])
        
        #model_name = './models/pred_model_' + str(m) + '.h5'
        #if debug_mode:
        #    model_name += '_debug'
        
        #model.save(model_name)
    
    #best_model = np.argsort(models_means)[-1]
    best_model = np.argmin(models_train_means)
    print('Using best model: Model', best_model + 1)
    print('Best Model Results:')
    print('Training Avg Diff:', models_train_means[best_model])
    print('Training Avg Diff Uncertainty :', models_train_stds[best_model])
    print('Testing Avg Diff:', models_test_means[best_model])
    print('Testing Avg Diff Uncertainty:', models_test_stds[best_model])
    print('\n')
    model = models[best_model]
    model.save('./models/' + model_name + '.h5')
    np.savez('./models/etc/' + model_name + '_data_indices', train_index=train_indices[best_model],
             test_index=test_indices[best_model], blind_index = blind_indices[best_model])
    
    return

# Just copied from regressor. Edit and implement classification
def classify_Polarity_Model(model_name, train_dir, seismos_train, arrivals_train, 
                    batch_size, epochs, model_iters, debug_mode=False):
    train_dir = check_String(train_dir)
    seismos_train = check_String(seismos_train)
    arrivals_train = check_String(arrivals_train)
    
    seismograms = np.load(train_dir + seismos_train)
    arrivals = np.load(train_dir + arrivals_train)
    
    if debug_mode:
        epochs=1
        model_iters=1
    
    models = []
    models_train_means = []
    models_train_stds = []
    models_test_means = []
    models_test_stds = []
    train_indices = []
    test_indices = []
    blind_indices = []
    for m in range(model_iters):
        
        data = sort_Data(seismograms, arrivals, debug_mode=debug_mode)
        
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
        
        model.compile(loss=huber_loss,
                      optimizer=Adam(),
                      metrics=[abs_Error])
        
        model.fit(data['train_x'], data['train_y'],
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,)
        
        train_pred = model.predict(data['train_x'])
        test_pred = model.predict(data['test_x'])
        
        model_train_diff = np.abs(data['train_y'] - train_pred)
        model_test_diff = np.abs(data['test_y'] - test_pred)
        model_train_mean = np.mean(model_train_diff)
        model_train_std = np.std(model_train_diff)
        model_test_mean = np.mean(model_test_diff)
        model_test_std = np.std(model_test_diff)
        
        print('Train Error:', model_train_mean, '+/-', model_train_std)
        print('Test Error:', model_test_mean, '+/-', model_test_std)
        
        models.append(model)
        models_train_means.append(model_train_mean)
        models_train_stds.append(model_train_std)
        models_test_means.append(model_test_mean)
        models_test_stds.append(model_test_std)
        train_indices.append(data['train_index'])
        test_indices.append(data['test_index'])
        blind_indices.append(data['blind_index'])
        
        #model_name = './models/pred_model_' + str(m) + '.h5'
        #if debug_mode:
        #    model_name += '_debug'
        
        #model.save(model_name)
    
    #best_model = np.argsort(models_means)[-1]
    best_model = np.argmin(models_train_means)
    print('Using best model: Model', best_model + 1)
    print('Best Model Results:')
    print('Training Avg Diff:', models_train_means[best_model])
    print('Training Avg Diff Uncertainty :', models_train_stds[best_model])
    print('Testing Avg Diff:', models_test_means[best_model])
    print('Testing Avg Diff Uncertainty:', models_test_stds[best_model])
    print('\n')
    model = models[best_model]
    model.save('./models/' + model_name + '.h5')
    np.savez('./models/etc/' + model_name + '_data_indices', train_index=train_indices[best_model],
             test_index=train_indices[best_model], blind_index = blind_indices[best_model])
    
    return
    
    