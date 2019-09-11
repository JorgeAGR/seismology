#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 22:35:00 2018

@author: jorgeagr
"""

import os
import numpy as np
from sac2npy import make_Arrays
from make_pred import predict_Arrival
from aux_funcs import read_Config, make_Dirs
from write_pred import write_Pred
import csv
from keras.models import load_model
import keras.losses
import keras.metrics
from tensorflow.losses import huber_loss

#temp line to use models... remove trace of abs_error until it works?
from models import abs_Error

make_Dirs()

config_dic = read_Config('./config/pred_config.txt')

print('Loading model...')
try:
    keras.losses.huber_loss = huber_loss
    keras.metrics.abs_error = abs_Error # temp line
    print(config_dic['model_name'])
    arrive_model = load_model('./models/'+ config_dic['model_name'] +'.h5')
    
except Exception as err:
    print(err)
    print('Error! Use an existing model or create a new one.')
    quit()

for n, d in enumerate(config_dic['pred_dir']):
    # Looks for files in pred_dir directories, makes them into NumPy arrays for prediction of models
    print('Working on directory:', d)
    print('Making SAC files into arrays...')
    make_Arrays(d, config_dic['arrival_var'])
    print('Predicting...')
    files, pred_avg, pred_err, flipped = predict_Arrival(arrive_model, d)
    print('Writing...')
    write_Pred(d, files, pred_avg, flipped)
    
    
    
    
    #file = './pred_data/seismograms_' + d.split('/')[-2] + '.npy'
    
    #name = d.split('/')[-2]
    # Predicts using the arrays made before, and saves predictions alongisde name of each file in a csv file
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