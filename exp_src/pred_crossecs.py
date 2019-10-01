#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 10:34:06 2019

@author: jorgeagr
"""

import numpy as np
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

def cut_Window(cross_sec, times, t_i, t_f):
    init = np.where(times == np.round(t_i, 1))[0][0]
    end = np.where(times == np.round(t_f, 1))[0][0]
    
    return cross_sec[init:end]

def shift_Max(seis, pred_var):
    data = seis.data
    time = seis.times()
    arrival = 0
    new_arrival = seis.stats.sac[pred_var]
    #for i in range(3):
    while (new_arrival - arrival) != 0:
        arrival = new_arrival
        init = np.where(time > (arrival - 1))[0][0]
        end = np.where(time > (arrival + 1))[0][0]
        
        amp_max = np.argmax(np.abs(data[init:end]))
        time_ind = np.arange(init, end, 1)[amp_max]
        
        new_arrival = time[time_ind]
        #print(new_arr)
        #if np.abs(new_arr - arrival) < 1e-2:
        #    break
    return arrival

keras.losses.huber_loss = huber_loss
pos_model = load_model('../auto_seismo/models/arrival_SS_pos_model_0025.h5')
neg_model = load_model('../auto_seismo/models/arrival_SS_neg_model_0025.h5')

# Picked by Lauren
cs = obspy.read('../../seismograms/cross_secs/5caps_wig/0.087_3.96.sac') # good
#cs = obspy.read('../../seismograms/cross_secs/5caps_wig/0.0785_3.74.sac') # meh
#cs = obspy.read('../../seismograms/cross_secs/5caps_wig/0.523_1.17.sac') # bad

# Randomly picked
#cs = obspy.read('../../seismograms/cross_secs/5caps_wig/0.087_0.54.sac')
#cs = obspy.read('../../seismograms/cross_secs/5caps_wig/0.087_2.70.sac')

cs = cs[0].resample(10)
times = cs.times()

shift = -cs.stats.sac.b
b = cs.stats.sac.b + shift
e = cs.stats.sac.e + shift

time_i_grid = np.arange(0, shift - 25 + 0.1, 0.1)
time_f_grid = np.arange(25, shift + 0.1, 0.1)

window_preds = np.zeros(len(time_i_grid))
window_shifted = np.zeros(len(time_i_grid))
for i, t_i, t_f in zip(range(len(time_i_grid)), time_i_grid, time_f_grid):
    if t_f > shift:
        break
    cs_window = cut_Window(cs, times, t_i, t_f)
    cs_window = cs_window / np.abs(cs_window).max()
    # Take the absolute value of the prediction to remove any wonky behavior in finding the max
    # Doesn't matter since they are bad predictions anyways
    cs.stats.sac.t6 = np.abs(pos_model.predict(cs_window.reshape(1, len(cs_window), 1))[0][0]) + t_i
    window_preds[i] += cs.stats.sac.t6#shift_Max(cs, 't6')
    window_shifted[i] += shift_Max(cs, 't6')
    #break

window_negs = np.zeros(len(time_i_grid))
for i, t_i, t_f in zip(range(len(time_i_grid)), time_i_grid, time_f_grid):
    if t_f > shift:
        break
    cs_window = cut_Window(cs, times, t_i, t_f)
    cs_window = cs_window / np.abs(cs_window).max()
    # Take the absolute value of the prediction to remove any wonky behavior in finding the max
    # Doesn't matter since they are bad predictions anyways
    cs.stats.sac.t6 = np.abs(neg_model.predict(cs_window.reshape(1, len(cs_window), 1))[0][0]) + t_i
    window_negs[i] += cs.stats.sac.t6#shift_Max(cs, 't6')
    #break

arrivals_pos, counts_pos = np.unique(np.round(window_preds, 0), return_counts=True)
arrivals_neg, counts_neg = np.unique(np.round(window_negs, 0), return_counts=True)
plateus = np.where(np.round(np.diff(window_preds),1) == 0)[0] + 1
#arrivals, counts = arrivals[counts > 50], counts[counts > 50]

fig, ax = plt.subplots()
ax.plot(time_i_grid, window_preds, color='black')
ax.plot(time_i_grid, window_negs, '.', color='black')
#ax.plot(time_i_grid[plateus], window_preds[plateus], '.', color='red')
ax.set_xlabel('Starting time [s]')
ax.set_ylabel('Predicted arrival [s]')

fig, ax = plt.subplots()
ax.plot(times, cs.data, color='black')
for ar in arrivals_pos[np.argsort(counts_pos)][-3:]:
    ax.axvline(ar, color='blue', linestyle='--')
for ar in arrivals_neg[np.argsort(counts_neg)][-3:]:
    ax.axvline(ar, color='red', linestyle='--')
ax.set_ylim(cs.data.min(), cs.data.max())
ax.set_xlim(times.min(), times.max())
ax.set_xlabel('Time [s]')
ax.set_ylabel('Amplitude')