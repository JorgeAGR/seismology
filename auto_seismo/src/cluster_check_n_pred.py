#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 18:11:29 2019

@author: jorgeagr
"""

'''
Temporary script to train models and compare prediction methods in the
NMSU cluster (doesn't have all the necessary libraries available.)
'''

import os
import numpy as np
#import sac2npy
from cluster_models import pred_Time_Model
from make_pred import predict_Arrival, predict_Train_Data
from aux_funcs import read_Config, make_Dirs

make_Dirs()

config_dic = read_Config('./cnn_config.txt')

# Trains a new set of models and picks the best one
print('Training model...')
arrive_model = pred_Time_Model(config_dic['train_dir'], 
                                 config_dic['seismos_train'],
                                 config_dic['arrivals_train'],
                                 config_dic['batch_size'], 
                                 config_dic['epochs'],
                                 config_dic['model_iters'],
                                 config_dic['debug_mode'])

preddir = '/SS_kept/'

files, pred_avg, pred_err, flipped = predict_Arrival(arrive_model, preddir, config_dic['debug_mode'])
print('Shifting Window Pred:', np.mean(pred_avg), '+\-', np.std(pred_avg))
np.savez('../train_data_shift_pred', files=files, pred_avg=pred_avg, 
         pred_err=pred_err, flipped=flipped)

simple_files, simple_pred = predict_Train_Data(arrive_model, preddir, config_dic['debug_mode'])
print('Simple Pred:', np.mean(simple_pred), '+\-', np.std(simple_pred))
np.savez('../train_data_simple_pred', files=simple_files, pred=simple_pred)