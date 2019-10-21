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

def find_Precursors(file_dir, cs_file, model):
    cs = obspy.read(file_dir+cs_file)
        
    cs = cs[0].resample(resample_Hz)
    times = cs.times()
    shift = -cs.stats.sac.b
    
    begin_time = -np.abs(-270) # Seconds before main arrival. Will become an input.
    begin_time = np.round(begin_time + shift, decimals=1)
    end_time = -np.abs(-80) # ditto above
    end_time = np.round(end_time + shift, decimals=1)
    
    time_i_grid = np.arange(begin_time, end_time - time_window + 0.1, 0.1)
    time_f_grid = np.arange(begin_time + time_window, end_time + 0.1, 0.1)
    window_preds = np.zeros(len(time_i_grid))
    #window_shifted = np.zeros(len(time_i_grid))
    print('Predicting...', end=' ')
    for i, t_i, t_f in zip(range(len(time_i_grid)), time_i_grid, time_f_grid):
        if t_f > shift:
            break
        cs_window = cut_Window(cs, times, t_i, t_f)
        cs_window = cs_window / np.abs(cs_window).max()
        # Take the absolute value of the prediction to remove any wonky behavior in finding the max
        # Doesn't matter since they are bad predictions anyways
        cs.stats.sac.t6 = np.abs(model.predict(cs_window.reshape(1, len(cs_window), 1))[0][0]) + t_i
        window_preds[i] += cs.stats.sac.t6
        # Ignoring shifted window for now
        #window_shifted[i] += shift_Max(cs, 't6')
    
    print('Finding arrivals...', end=' ')
    dbscan = DBSCAN(eps=0.05, min_samples=2)
    dbscan.fit(window_preds.reshape(-1,1))
    clusters, counts = np.unique(dbscan.labels_, return_counts=True)
    if -1 in clusters:
        clusters = clusters[1:]
        counts = counts[1:]
    relevant_preds = 2
    discont_ind = np.argsort(counts)[-relevant_preds:]
    clusters = clusters[discont_ind]
    counts = counts[discont_ind]
    arrivals_pos = np.zeros(relevant_preds)
    arrivals_pos_err = np.zeros(relevant_preds)
    arrivals_quality = np.zeros(relevant_preds)
    for i, c in enumerate(clusters):
        arrivals_pos[i] = np.mean(window_preds[dbscan.labels_ == c])
        arrivals_pos_err[i] = np.std(window_preds[dbscan.labels_ == c])
        arrivals_quality[i] = counts[i] / n_preds
    disc_660, disc_410 = np.argsort(arrivals_pos)
    
    print('Finding amplitudes...')
    init410 = np.where(times < arrivals_pos[disc_410])[0][-1]
    init660 = np.where(times < arrivals_pos[disc_660])[0][-1]
    amp410 = cs.data[init410:init410+2].max()
    amp660 = cs.data[init660:init660+2].max()
    
    arrivals_pos = arrivals_pos - shift
    string_410 = str(arrivals_pos[disc_410]) + ',' + str(arrivals_pos_err[disc_410]) + ',' + str(amp410) + ',' + str(arrivals_quality[disc_410])
    string_660 = str(arrivals_pos[disc_660]) + ',' + str(arrivals_pos_err[disc_660]) + ',' + str(amp660) + ',' + str(arrivals_quality[disc_660])
    return string_410, string_660
    
keras.losses.huber_loss = huber_loss
pos_model = load_model('../auto_seismo/models/arrival_SS_pos_model_0040.h5')
#neg_model = load_model('../auto_seismo/models/arrival_SS_neg_model_0040.h5')
resample_Hz = 10
time_window = 40
n_preds = time_window * resample_Hz # Maximum number of times the peak could be found, from sliding the window

file_dir = '../../seismograms/cross_secs/5caps_wig/'
files = np.sort([f for f in os.listdir(file_dir) if '.sac' in f])
with open(file_dir.split('/')[-2] + '_preds.csv', 'w+') as pred_csv:
    print('file,410pred,410err,410amp,410qual,660pred,660err,660amp,660qual', file=pred_csv)

for f, cs_file in enumerate(files):
    print('File', f+1, '/', len(files),'...', end=' ')
    string_410, string_660 = find_Precursors(file_dir, cs_file, pos_model)
    with open(file_dir.split('/')[-2] + '_preds.csv', 'a') as pred_csv:
        print(cs_file + ',' + string_410 + ',' + string_660, file=pred_csv)