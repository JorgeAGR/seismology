#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 11:11:12 2019

@author: jorgeagr
"""

import os
import numpy as np
from models import pred_Time_Model
from aux_funcs import read_Config, make_Dirs
import tensorflow as tf
from keras.backend import tensorflow_backend as tfb

# Checks for the necessary directories and makes them if not present
make_Dirs()

# Read in config files for tensorflow and models settings
tf_config = read_Config('./config/tf_config.txt')
pos_config = read_Config('./config/pos_model_config.txt')
neg_config = read_Config('./config/neg_model_config.txt')

models = (pos_config, neg_config)

# Execute the model training
if not tf_config['gpu']:
    parallel_config=tf.ConfigProto(intra_op_parallelism_threads=tf_config['nodes']*tf_config['threads'])
    with tf.Session(config=parallel_config) as sess:
        tfb.set_session(sess)
        for model in models:
            print('Creating new model:', model['model_name'])
            if model['model_type'] == 'regression':
                if model['debug_mode']:
                    model['model_name'] += '_debug'
                
                pred_Time_Model(model)
                
else:
    for model in models:
            print('Creating new model:', model['model_name'])
            if model['model_type'] == 'regression':
                if model['debug_mode']:
                    model['model_name'] += '_debug'
                
                pred_Time_Model(model)