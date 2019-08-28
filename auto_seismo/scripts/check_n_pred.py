#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 22:35:00 2018

@author: jorgeagr
"""

import os
import numpy as np
#import sac2npy
from seismo_arrays import make_arrays
from arrival_pred import init_Arrive_Model
from make_pred import predict_arrival
from read_config import read_config
import csv
from keras.models import load_model
import keras.losses
import tensorflow as tf


config_dic = read_config('./cnn_config.txt')
'''
# Load cnn_config.txt file and reads the set values for variables to be used
config_dic = {}
with open('./cnn_config.txt') as config:
    lines = config.readlines()
    for i, line in enumerate(lines):
        if (line != '\n') and ('#' not in line[0]):
            split_vals = False
            if ',' in line:
                split_vals = True
            line = line.rstrip('\n').split(':')
            config_dic[line[0]] = [text for text in line[1:]]
            if split_vals:
                config_dic[line[0]] = config_dic[line[0]][0].split(',')
            if (len(config_dic[line[0]]) == 1) and (line[0] != 'pred_dir'):
                config_dic[line[0]] = config_dic[line[0]][0]
            try:
                config_dic[line[0]] = int(config_dic[line[0]])
            except:
                pass
'''
# Trains a new model if desired or if none present, else load the present one
if config_dic['debug_mode'] == 1:
    print('Using debug mode...')
    arrive_model = init_Arrive_Model(config_dic['train_dir'], 
                                     config_dic['seismos_train'],
                                     config_dic['arrivals_train'],
                                     128, 1, 1, debug_mode=True)
    
elif (config_dic['new_model'] == 1) or (len(os.listdir('./models/')) == 0):
    print('Creating new model...')
    arrive_model = init_Arrive_Model(config_dic['train_dir'], 
                                     config_dic['seismos_train'],
                                     config_dic['arrivals_train'],
                                     config_dic['batch_size'],
                                     config_dic['epochs'],
                                     config_dic['model_iters'])

    arrive_model.save('./models/arrival_prediction_model.h5')
    
else:
    print('Loading model...')
    try:
        keras.losses.huber_loss = tf.losses.huber_loss
        arrive_model = load_model('./models/arrival_prediction_model.h5')
    except:
        print('Error! Creating new model...')
        arrive_model = init_Arrive_Model(config_dic['train_dir'], 
                                         config_dic['seismos_train'],
                                         config_dic['arrivals_train'],
                                         config_dic['batch_size'], 
                                         config_dic['epochs'],
                                         config_dic['model_iters'])

        arrive_model.save('./models/arrival_prediction_model.h5')

for n, d in enumerate(config_dic['pred_dir']):
    # Looks for files in pred_dir directories, makes them into NumPy arrays for prediction of models
    print('Working on directory:', d)
    print('Making SAC files into arrays...')
    make_arrays(d, config_dic['arrival_var'])
    files, pred_avg, pred_err = predict_arrival(arrive_model, d)
    
    #file = './pred_data/seismograms_' + d.split('/')[-2] + '.npy'
    
    #name = d.split('/')[-2]
    # Predicts using the arrays made before, and saves predictions alongisde name of each file in a csv file
    print('Predicting...')
    #pred = arrive_model.predict(np.load(file)).flatten()
    #arrival_times = np.load('./pred_data/cut_times_' + name + '.npy') + pred
    '''
    files = np.load('./results/file_names_' + name + '.npy')
    with open('./results/results_' + name + '.csv', 'w+') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')    
        for i in range(len(pred)):
            writer.writerow([files[i], arrival_times[i]])
    os.remove('./results/file_names_' + name + '.npy')
    print('CSV saved')
    '''