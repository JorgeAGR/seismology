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
from keras.models import load_model
import keras.losses
import keras.metrics
from tensorflow.losses import huber_loss
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from sklearn.cluster import DBSCAN

height = 16
width = 25

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
pos_model = load_model('../auto_seismo/models/arrival_SS_pos_model_0040.h5')
neg_model = load_model('../auto_seismo/models/arrival_SS_neg_model_0040.h5')
time_window = 40

# Picked by Lauren
#cs = obspy.read('../../seismograms/cross_secs/5caps_wig/0.087_3.96.sac') # good
#cs = obspy.read('../../seismograms/cross_secs/5caps_wig/0.785_3.74.sac') # meh
#cs = obspy.read('../../seismograms/cross_secs/5caps_wig/0.523_1.17.sac') # bad

# Randomly picked
#cs = obspy.read('../../seismograms/cross_secs/5caps_wig/0.087_0.54.sac')
#cs = obspy.read('../../seismograms/cross_secs/5caps_wig/0.087_2.70.sac')

cap = '5'
file = '0.174_1.71'
cs = obspy.read('../../seismograms/cross_secs/'+cap+'caps_wig/'+file+'.sac')

cs = cs[0].resample(10)
times = cs.times()

shift = -cs.stats.sac.b
begin_time = 80 # will be user input
end_time = 260 # ditto

time_i_grid = np.arange(begin_time, end_time - time_window + 0.1, 0.1)
time_f_grid = np.arange(begin_time + time_window, end_time + 0.1, 0.1)

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
'''
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
'''
#arrivals_pos, counts_pos = np.unique(np.round(window_preds, 0), return_counts=True)
#arrivals_neg, counts_neg = np.unique(np.round(window_negs, 0), return_counts=True)
#plateus = np.where(np.diff(np.round(window_preds,1)) == 0)[0] + 1
#arrivals, counts = arrivals[counts > 50], counts[counts > 50]

# Can think of max distance eps as the max allowed variance??
# DBSCAN looks for dense clusters, 
#while True:
min_samples = 2#2*time_window# / 2 / 0.1
dbscan = DBSCAN(eps=0.05, min_samples=min_samples)
dbscan.fit(window_preds.reshape(-1,1))
#n_clusters = len(set(dbscan.labels_)) - (1 if -1 in dbscan.labels_ else 0)
clusters, counts_pos = np.unique(dbscan.labels_, return_counts=True)
if -1 in clusters:
    clusters = clusters[1:]
    counts_pos = counts_pos[1:]
#    if len(clusters) > 2:
#        break
#    else:
#        min_samples += -10
sorted_ind = np.argsort(counts_pos)[-2:]
arrivals_pos = np.zeros(len(clusters))
arrival_appearance = np.zeros(len(clusters))
for c in clusters:
    arrivals_pos[c] = np.mean(window_preds[dbscan.labels_ == c])
    arrival_appearance[c] = counts_pos[c] / 400

'''
# Difference between predictions at t+1 and t
preds_diff = np.diff(window_preds)
# Which ones are close 0, to find plateus
zeros = np.isclose(preds_diff, 0, atol=0.05)
# Calculate the difference in indeces between the zeros.
# Where jumps == 1, means they belong in the same plateu
# Anything larger, that is a jump
pred_ind = np.arange(0, len(preds_diff), 1)
pred_ind[~zeros] = 0
jumps = np.zeros(len(preds_diff))
jumps[1:] = np.abs(np.diff(pred_ind))
jumps[jumps == 1] = 1
jumps[jumps > 1] = 0

fig, ax = plt.subplots()
ax.plot(time_i_grid[1:], preds_diff)
ax.plot(time_i_grid[1:][zeros], preds_diff[zeros], '.')
ax.plot(time_i_grid[1:], jumps)
ax.set_ylim(-5, 10)
'''
fig, ax = plt.subplots()
ax.plot(time_i_grid, window_preds, '.', color='black')
for cluster in clusters:#[sorted_ind]:
    ax.plot(time_i_grid[dbscan.labels_ == cluster], window_preds[dbscan.labels_ == cluster], '.')
#ax.plot(time_i_grid, window_negs, '.', color='black')
#ax.plot(time_i_grid[plateus], window_preds[plateus], '.', color='red')
ax.set_xlabel('Starting time [s]')
ax.set_ylabel('Predicted arrival [s]')
ax.yaxis.set_minor_locator(mtick.MultipleLocator(10))
ax.grid()

cs_norm = cs.data / np.abs(cs.data).max()
fig, ax = plt.subplots()
ax.plot(times, cs_norm, color='black')
for i, ar in enumerate(arrivals_pos[np.argsort(counts_pos)][-2:]):
    ax.axvline(ar, color='blue', linestyle='--')
    ax.text(ar-5, 0.1, np.sort(counts_pos)[-2:][i], rotation=90, fontsize=16)
ax.axvline(ar, color='blue', linestyle='--', label='positive model')
#for i, ar in enumerate(arrivals_neg[np.argsort(counts_neg)][-5:]):
#   ax.axvline(ar, color='red', linestyle='--')
#   ax.text(ar-5, 0.1, np.sort(counts_neg)[-5:][i], rotation=90, fontsize=16)
#ax.axvline(ar, color='red', linestyle='--', label='negative model')
ax.set_ylim(cs_norm.min(), cs_norm.max())
ax.set_xlim(times.min(), times.max())
ax.set_xlabel('Time [s]')
ax.set_ylabel('Amplitude')
#ax.set_title('5caps_wig/0.087_3.96')
ax.xaxis.set_minor_locator(mtick.MultipleLocator(10))
ax.legend(loc='upper right')
fig.tight_layout()