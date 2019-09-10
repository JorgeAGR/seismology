#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 11:54:15 2019

@author: jorgeagr
"""

from sac2npy import training_Arrays
from aux_funcs import make_Dirs, read_Config

make_Dirs()

train_config = read_Config('./config/train_data_config.txt')

training_Arrays(train_config['datadir'], 'good', train_config['th_arrival'], 
                train_config['arrival_var'], 1, train_config['type'])