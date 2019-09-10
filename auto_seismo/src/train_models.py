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

make_Dirs()

tf_config = read_Config('./config/tf_config.txt')
pos_config = read_Config('./config/pos_model_config.txt')
neg_config = read_Config('./config/neg_model_config.txt')

models = (pos_config, neg_config)

if not tf_config['gpu']:
    with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=tf_config['nodes']*tf_config['threads'])) as sess:
        tfb.set_session(sess)
        for model in models:
            print('Creating new model:', model['model_name'])
            if model['model_type'] == 'regression':
                if model['debug_mode']:
                    model['model_name'] += '_debug'
                
                pred_Time_Model(model['model_name'],
                                model['train_dir'], 
                                model['seismos_train'],
                                model['arrivals_train'],
                                model['batch_size'],
                                model['epochs'],
                                model['model_iters'],
                                debug_mode=model['debug_mode'])
                
else:
    for model in models:
            print('Creating new model:', model['model_name'])
            if model['model_type'] == 'regression':
                if model['debug_mode']:
                    model['model_name'] += '_debug'
                
                pred_Time_Model(model['model_name'],
                                model['train_dir'], 
                                model['seismos_train'],
                                model['arrivals_train'],
                                model['batch_size'],
                                model['epochs'],
                                model['model_iters'],
                                debug_mode=model['debug_mode'])