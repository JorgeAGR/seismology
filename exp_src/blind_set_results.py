# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 23:58:26 2019

@author: jorge
"""
import os
import obspy
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from keras.models import load_model
import keras.losses
import keras.metrics
from tensorflow.losses import huber_loss
import keras.backend as tfb

def abs_Error(y_true, y_pred):
    return tfb.mean(tfb.abs(y_true - y_pred))

golden_ratio = (np.sqrt(5) + 1) / 2
width = 12
height = width / golden_ratio

mpl.rcParams['figure.figsize'] = (width, height)
mpl.rcParams['font.size'] = 28
mpl.rcParams['figure.titlesize'] = 'large'
mpl.rcParams['legend.fontsize'] = 'small'
mpl.rcParams['xtick.major.size'] = 12
mpl.rcParams['xtick.minor.size'] = 8
mpl.rcParams['xtick.labelsize'] = 24
mpl.rcParams['ytick.major.size'] = 12
mpl.rcParams['ytick.minor.size'] = 8
mpl.rcParams['ytick.labelsize'] = 24

keras.losses.huber_loss = huber_loss
keras.metrics.abs_error = abs_Error
pos_model = load_model('./models/arrival_SS_pos_model.h5')
neg_model = load_model('./models/arrival_SS_neg_model.h5')

pos_model_ind = np.load('./models/etc/arrival_SS_pos_model_data_indices.npz')
neg_model_ind = np.load('./models/etc/arrival_SS_neg_model_data_indices.npz')

seismograms = np.load('./train_data/seismograms_SS.npy')