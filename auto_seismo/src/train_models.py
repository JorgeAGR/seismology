#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 11:11:12 2019

@author: jorgeagr
"""

import os
import numpy as np
from models import pred_Time_Model
from aux_funcs import read_config, make_dirs

make_dirs()

pos_config = read_config('./config/pos_model_config.txt')
neg_config = read_config('./config/neg_model_config.txt')

models = (pos_config, neg_config)

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