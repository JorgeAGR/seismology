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

# Checks for the necessary directories and makes them if not present
make_Dirs()

# Load config file for prediction variables
config_dic = read_Config('./config/pred_config.txt')

print('Loading model...')
try:
    # Attempt to load the models, else raise an error 
    keras.losses.huber_loss = huber_loss
    models = []
    for m in config_dic['model_names']:
        print(m)
        models.append(load_model('./models/'+ m +'.h5'))
except Exception as err:
    print(err)
    print('Error! Use an existing model or create a new one.')
    quit()

for n, d in enumerate(config_dic['pred_dir']):
    print('Working on directory:', d)
    print('Making SAC files into arrays...')
    # Iterates through files in directory and writes them as npy arrays
    make_Arrays(d, config_dic['arrival_var'], config_dic['window_before'], config_dic['window_after'])
    print('Predicting...')
    # Predict arrival times for all seismograms in the directory
    files, pred_avg, pred_err, flipped = predict_Arrival(d, models)
    # Write the predictions to the SAC files
    print('Writing...')
    write_Pred(d, files, pred_avg, flipped, config_dic['prediction_var'])