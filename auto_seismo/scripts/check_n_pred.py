#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 22:35:00 2018

@author: jorgeagr
"""

import os
import numpy as np
#import sac2npy
import seismo_arrays
import arrival_pred
import csv
from keras.models import load_model

# Load cnn_config.txt file and reads the set values for variables to be used
config_dic = {}
with open('./cnn_config.txt') as config:
    lines = config.readlines()
    for i, line in enumerate(lines):
        if (line != '\n') and ('#' not in line[0]):
            split_vals = False
            if '||' in line:
                split_vals = True
            line = line.rstrip('\n').split(':')
            config_dic[line[0]] = [text for text in line[1:]]
            if split_vals:
                config_dic[line[0]] = config_dic[line[0]][0].split('||')
            if (len(config_dic[line[0]]) == 1) and (line[0] != 'pred_dir'):
                config_dic[line[0]] = config_dic[line[0]][0]
            try:
                config_dic[line[0]] = int(config_dic[line[0]])
            except:
                pass

# Trains a new model if desired or if none present, else load the present one
if config_dic['debug_mode'] == 1:
    print('Using debug mode...')
    arrive_model = arrival_pred.init_Arrive_Model(config_dic['train_dir'], 
                                                  config_dic['seismos_train'],
                                                  config_dic['arrivals_train'],
                                                  128, 1, 1, debug_mode=True)
    
elif (config_dic['new_model'] == 1) or (len(os.listdir('./models/')) == 0):
    print('Creating new model...')
    arrive_model = arrival_pred.init_Arrive_Model(config_dic['train_dir'], 
                                                  config_dic['seismos_train'],
                                                  config_dic['arrivals_train'],
                                                  config_dic['batch_size'], 
                                                  config_dic['epochs'],
                                                  config_dic['model_iters'])

    arrive_model.save('./models/arrival_prediction_model.h5')
    
else:
    print('Loading model...')
    arrive_model = load_model('./models/arrival_prediction_model.h5')

print('Making SAC files into arrays...')
for n, d in enumerate(config_dic['pred_dir']):
    # Looks for files in pred_dir directories, makes them into NumPy arrays for prediction of models
    print('File', n+1, '/', len(config_dic['pred_dir']))
    seismo_arrays.make_arrays(d, config_dic['arrival_var'])

# Finds all the arrays of seismograms to be predicted
pred_data_files = ['./pred_data/seismograms_' + d.split('/')[-2] + '.npy' for d in config_dic['pred_dir']]

for n, file in enumerate(pred_data_files):
    # Predicts using the arrays made before, and saves predictions alongisde name of each file in a csv file
    print('Predicting:', n+1, '/', len(pred_data_files), '...')
    pred = arrive_model.predict(np.load(file)).flatten()
    
    files = np.load('./results/file_names_' + file.split('/')[-1][12:])
    with open('./results/results_' + file.split('/')[-1][:-4] + '.csv', 'w+') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')    
        for i in range(len(pred)):
            writer.writerow([files[i], pred[i]])
    os.remove('./results/file_names_' + file.split('/')[-1][12:])
