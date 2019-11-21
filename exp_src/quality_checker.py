# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 23:26:33 2019

@author: jorge
"""

import os
import obspy
import numpy as np
import pandas as pd
from keras.models import load_model
import keras.losses
import keras.metrics
from tensorflow.losses import huber_loss
from sklearn.cluster import DBSCAN
keras.losses.huber_loss = huber_loss
import matplotlib.pyplot as plt

def cut_Window(seis, times, t_i, t_f):
    init = np.where(times == np.round(t_i, 1))[0][0]
    end = np.where(times == np.round(t_f, 1))[0][0]
    
    return seis[init:end]

resample_Hz = 10

datadir = '../../seismograms/SS_kept/'
model_dir = '../auto_seismo/models/'

model = load_model(model_dir+'arrival_SS_pos_model_0040.h5')
files = np.sort(os.listdir(datadir))

file = files[0]

seismogram = obspy.read(datadir + file)
seismogram = seismogram[0].resample(resample_Hz).detrend()

seis = seismogram.data

times = seismogram.times()
shift = -seismogram.stats.sac.b

begin_time = 0 # Seconds before main arrival. Will become an input.
end_time = 500
time_window = 40

time_i_grid = np.arange(begin_time, end_time - time_window + 1/resample_Hz, 1/resample_Hz)
time_f_grid = np.arange(begin_time + time_window, end_time + 1/resample_Hz, 1/resample_Hz)
window_preds = np.zeros(len(time_i_grid))
#window_shifted = np.zeros(len(time_i_grid))
print('Predicting...', end=' ')
for i, t_i, t_f in zip(range(len(time_i_grid)), time_i_grid, time_f_grid):
    s_window = cut_Window(seis, times, t_i, t_f)
    s_window = s_window / np.abs(s_window).max()
    # Take the absolute value of the prediction to remove any wonky behavior in finding the max
    # Doesn't matter since they are bad predictions anyways
    seismogram.stats.sac.t7 = np.abs(model.predict(s_window.reshape(1, len(s_window), 1))[0][0]) + t_i
    window_preds[i] += seismogram.stats.sac.t7
    # Ignoring shifted window for now
    #window_shifted[i] += shift_Max(cs, 't6')
