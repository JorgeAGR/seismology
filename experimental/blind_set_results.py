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
from src_funcs import abs_Error, predict_Arrival

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
pos_model = load_model('../auto_seismo/models/arrival_SS_pos_model_1030.h5')
neg_model = load_model('../auto_seismo/models/arrival_SS_neg_model_1030.h5')

pos_model_ind = np.load('../auto_seismo/models/etc/arrival_SS_pos_model_1030_data_indices.npz')
neg_model_ind = np.load('../auto_seismo/models/etc/arrival_SS_neg_model_1030_data_indices.npz')

seismograms = np.load('../auto_seismo/train_data/seismograms_SS_1030.npy')
seismograms_flip = np.load('../auto_seismo/train_data/seismograms_flipped_SS_1030.npy')
arrivals = np.load('../auto_seismo/train_data/arrivals_SS_1030.npy')

blind_pos_seis = seismograms[pos_model_ind['blind_index']].reshape(len(pos_model_ind['blind_index']), len(seismograms[0]), 1)
blind_neg_seis = seismograms_flip[neg_model_ind['blind_index']].reshape(len(neg_model_ind['blind_index']), len(seismograms[0]), 1)

pos_simple_pred = pos_model.predict(blind_pos_seis)
neg_simple_pred = neg_model.predict(blind_neg_seis)

pos_simple_err = pos_simple_pred - arrivals[pos_model_ind['blind_index']]
neg_simple_err = neg_simple_pred - arrivals[neg_model_ind['blind_index']]

files = np.load('../auto_seismo/train_data/file_names_SS_1030.npy')
pos_files = files[pos_model_ind['blind_index']]
neg_files = files[neg_model_ind['blind_index']]

pos_shift_pred = predict_Arrival('/SS_kept/', files=pos_files)
neg_shift_pred = predict_Arrival('/SS_kept/', files=neg_files)
