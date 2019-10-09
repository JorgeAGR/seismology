#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 20:32:16 2019

@author: jorgeagr
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
from keras.models import Model

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
pos_model = load_model('../auto_seismo/models/arrival_SS_pos_model_0040.h5')

layer_outputs = [layer.output for layer in pos_model.layers[:-5]]
layer_names = [layer.name for layer in pos_model.layers[:-5]]

activation_model = Model(inputs=pos_model.input, outputs=layer_outputs)

file = '../auto_seismo/pred_data/seis_1/19910921A.tonga.AQU.BHT.npz'
file_npz = np.load(file)
seis = file_npz['noflips'][0]
seis = seis.reshape(1, len(seis), 1)

features = activation_model.predict(seis)

fig, ax = plt.subplots(nrows=10, sharex=True)
ax[0].set_title(layer_names[0])
for j in range(10):
    eg_feature = features[0][:,:,j][0]
    ax[j].plot(eg_feature)
plt.tight_layout