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

keras.losses.huber_loss = huber_loss
pos_model = load_model('../auto_seismo/models/arrival_SS_pos_model_0040.h5')
neg_model = load_model('../auto_seismo/models/arrival_SS_neg_model_0040.h5')
time_window = 40

file_dir = '../../seismograms/cross_secs/5caps_wig/'
files = np.sort(os.listdir(file_dir))[:4]

with open(file_dir.split('/')[-2] + '_preds.csv', 'w+') as pred_csv:
    print('file,410pred,410err,660pred,660err', file=pred_csv)
    for cs_file in files:
        cs = obspy.read(file_dir+cs_file)
        
        cs = cs[0].resample(10)
        times = cs.times()
        
        shift = -cs.stats.sac.b
        b = cs.stats.sac.b + shift
        e = cs.stats.sac.e + shift
        
        time_i_grid = np.arange(0, shift - time_window + 0.1, 0.1)
        time_f_grid = np.arange(time_window, shift + 0.1, 0.1)
        
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
            window_preds[i] += cs.stats.sac.t6
            window_shifted[i] += shift_Max(cs, 't6')
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
        '''
        # Can think of max distance eps as the max allowed variance??
        # DBSCAN looks for dense clusters, 
        min_samples = 2*time_window# / 2 / 0.1
        dbscan = DBSCAN(eps=0.5, min_samples=min_samples)
        dbscan.fit(window_preds.reshape(-1,1))
        clusters, counts_pos = np.unique(dbscan.labels_, return_counts=True)
        if -1 in clusters:
            clusters = clusters[1:]
            counts_pos = counts_pos[1:]
        arrivals_pos = np.zeros(len(clusters))
        arrivals_pos_err = np.zeros(len(clusters))
        for c in clusters:
            arrivals_pos[c] = np.mean(window_preds[dbscan.labels_ == c])
            arrivals_pos_err[c] = np.std(window_preds[dbscan.labels_ == c])
        discont_ind = np.argsort(counts_pos)[-2:]
        arrivals_pos = arrivals_pos[discont_ind] - shift
        arrivals_pos_err = arrivals_pos_err[discont_ind]
        counts_pos = counts_pos[discont_ind]
        
        disc_660, disc_410 = np.argsort(arrivals_pos)
        
        string_410 = str(arrivals_pos[disc_410]) + ',' + str(arrivals_pos_err[disc_410])
        string_660 = str(arrivals_pos[disc_660]) + ',' + str(arrivals_pos_err[disc_660])
        print(cs_file + ',' + string_410 + ',' + string_660, file=pred_csv)
