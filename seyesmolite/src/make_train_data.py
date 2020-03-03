#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 11:54:15 2019

@author: jorgeagr
"""

from sac2npy import training_Arrays
from aux_funcs import make_Dirs, read_Config

# Checks for the necessary directories and makes them if not present
make_Dirs()

# Load config file for the routine
train_config = read_Config('./config/train_data_config.txt')

# Execute creation of training and testing data
training_Arrays(train_config['datadir'], train_config['th_arrival'], 
                train_config['arrival_var'], 1, train_config['type'], 
                window_before=train_config['window_before'], window_after=train_config['window_after'])