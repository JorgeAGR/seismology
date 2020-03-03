#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 15:32:20 2020

@author: jorgeagr
"""

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
import time as timekeep

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

cap = '15'
file = 'n15_156.50'#'n45_232.87'
#cs = obspy.read('../../seismograms/cross_secs/'+cap+'caps_deg/'+file+'.sac')
tic = timekeep.time()
cs = obspy.read('../../seismograms/SS_kept/19880704A.peru.INU.BHT.s_fil')

cs = cs[0].resample(10)
times = cs.times()

shift = -cs.stats.sac.b
begin_time = -np.abs(-280) # Seconds before main arrival. Will become an input.
begin_time = np.round(begin_time + shift, decimals=1)
end_time = -np.abs(-80) # ditto above
end_time = np.round(end_time + shift, decimals=1)
time_i_grid = np.arange(begin_time, end_time - time_window, 0.1)
time_f_grid = np.arange(begin_time + time_window, end_time, 0.1)

window_preds = np.zeros(len(time_i_grid))
window_shifted = np.zeros(len(time_i_grid))
for i, t_i, t_f in zip(range(len(time_i_grid)), time_i_grid, time_f_grid):
    if t_f > shift:
        pass#break
    cs_window = cut_Window(cs, times, t_i, t_f)
    cs_window = cs_window / np.abs(cs_window).max()
    # Take the absolute value of the prediction to remove any wonky behavior in finding the max
    # Doesn't matter since they are bad predictions anyways
    cs.stats.sac.t6 = np.abs(pos_model.predict(cs_window.reshape(1, len(cs_window), 1))[0][0]) + t_i
    window_preds[i] += cs.stats.sac.t6#shift_Max(cs, 't6')
    #window_shifted[i] += shift_Max(cs, 't6')
    #break

window_negs = np.zeros(len(time_i_grid))
for i, t_i, t_f in zip(range(len(time_i_grid)), time_i_grid, time_f_grid):
    if t_f > shift:
        pass
    cs_window = -cut_Window(cs, times, t_i, t_f)
    cs_window = cs_window / np.abs(cs_window).max()
    # Take the absolute value of the prediction to remove any wonky behavior in finding the max
    # Doesn't matter since they are bad predictions anyways
    cs.stats.sac.t6 = np.abs(pos_model.predict(cs_window.reshape(1, len(cs_window), 1))[0][0]) + t_i
    window_negs[i] += cs.stats.sac.t6#shift_Max(cs, 't6')
    #break

def cluster_preds(predictions, eps=0.05, min_neighbors=2):
    dbscan = DBSCAN(eps, min_neighbors)
    dbscan.fit(predictions.reshape(-1,1))
    clusters, counts = np.unique(dbscan.labels_, return_counts=True)
    if -1 in clusters:
        clusters = clusters[1:]
        counts = counts[1:]
    arrivals = np.zeros(len(clusters))
    arrivals_qual = np.zeros(len(clusters))
    for c in clusters:
        arrivals[c] = np.mean(predictions[dbscan.labels_ ==  c])
        arrivals_qual[c] = counts[c]/400
    return arrivals, arrivals_qual

relevant_preds = 8
arrivals_pos, arrivals_pos_qual = cluster_preds(window_preds)
arrivals_neg, arrivals_neg_qual = cluster_preds(window_negs)

toc = timekeep.time()
print('Time to predict: {} seconds'.format(toc-tic))