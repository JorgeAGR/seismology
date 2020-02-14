#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 17:48:13 2020

@author: jorgeagr
"""

import os
import argparse

resample_Hz = 10
time_window = 40

# For prediction clustering
eps=5
percent_data=0.99

parser = argparse.ArgumentParser(description='Predict precursor arrivals in vespagram cross-sectional data.')
parser.add_argument('file_dir', help='Cross-section SAC files directory.', type=str)
parser.add_argument('phase', help='Phase to pick for.', type=str)
parser.add_argument('model_name', help='Path to model H5 file.', type=str)
args = parser.parse_args()

file_dir = args.file_dir
model_name = args.model_name
phase = args.phase

if file_dir[-1] != '/':
    file_dir += '/'

import obspy
import numpy as np
from scipy.interpolate import interp1d
import tensorflow as tf
from sklearn.cluster import DBSCAN
import time

def cut_Window(cross_sec, times, t_i, t_f):
    init = np.where(times == np.round(t_i, 1))[0][0]
    end = np.where(times == np.round(t_f, 1))[0][0]
    
    return cross_sec[init:end]

def shift_Max(seis, arrival):
    data = seis.data
    time = seis.times()
    init = np.where(time > (arrival - 1))[0][0]
    end = np.where(time > (arrival + 1))[0][0]
    
    # Interpolate to find "true" maximum
    f = interp1d(time[init:end], data[init:end], kind='cubic')
    t_grid = np.linspace(time[init], time[end-1], num=200)
    amp_max = np.argmax(np.abs(f(t_grid)))
    arrival = t_grid[amp_max]
    
    return arrival

def scan(seis, times, time_i_grid, time_f_grid, shift, model, negative=False):
    window_preds = np.zeros(len(time_i_grid))
    for i, t_i, t_f in zip(range(len(time_i_grid)), time_i_grid, time_f_grid):
        seis_window = cut_Window(seis, times, t_i, t_f) * (-1)**negative
        seis_window = seis_window / np.abs(seis_window).max()
        # Take the absolute value of the prediction to remove any wonky behavior in finding the max
        # Doesn't matter since they are bad predictions anyways
        window_preds[i] += np.abs(model.predict(seis_window.reshape(1, len(seis_window), 1))[0][0]) + t_i
    return window_preds

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

def pick_Phase(file_dir, seis_file, phase_name, model, store_header='auto', relevant_preds=1, window_size=40, sample_Hz=10):

    seis = obspy.read(file_dir+seis_file)
    seis = seis[0].resample(sample_Hz)
    times = seis.times()
    
    phases_in_seis = [seis.stats.sac[k].rstrip(' ') for k in seis.stats.sac.keys() if 'kt' in k]
    phases_headers = [k.lstrip('k') for k in seis.stats.sac.keys() if 'kt' in k]
    phase_var = dict(zip(phases_in_seis, phases_headers))[phase_name]
    
    shift = -seis.stats.sac.b
    begin_time = seis.stats.sac[phase_var] - window_size#seis.stats.sac.b
    begin_time = np.round(begin_time + shift, decimals=1)
    end_time = seis.stats.sac[phase_var] + 2.5*window_size#seis.stats.sac.e
    end_time = np.round(end_time + shift, decimals=1)

    time_i_grid = np.arange(begin_time, end_time - time_window, 1/sample_Hz)
    time_f_grid = np.arange(begin_time + time_window, end_time, 1/sample_Hz)

    pos_preds = scan(seis, times, time_i_grid, time_f_grid, shift, model)
    neg_preds = scan(seis, times, time_i_grid, time_f_grid, shift, model, negative=True)

    arrivals_pos, arrivals_pos_qual = cluster_preds(pos_preds)
    arrivals_neg, arrivals_neg_qual = cluster_preds(neg_preds)
    
    highest_pos_ind = np.argsort(arrivals_pos_qual)[-1]
    highest_neg_ind = np.argsort(arrivals_neg_qual)[-1]
    arrival_pos = arrivals_pos[highest_pos_ind]
    arrival_pos_qual = arrivals_pos_qual[highest_pos_ind]
    arrival_neg = arrivals_neg[highest_neg_ind]
    arrival_neg_qual = arrivals_neg_qual[highest_neg_ind]
    
    t_diff = arrival_pos - arrival_neg
    qual_diff = np.abs(arrival_pos_qual - arrival_neg_qual)
    # If theyre this close and of similar quality,
    # then the model is picking the side lobe.
    if (np.abs(t_diff) <= window_size) and (qual_diff < 0.1):
        if t_diff < 0:
            arrival = arrival_neg
            arrival_qual = arrival_neg_qual
        else:
            arrival = arrival_pos
            arrival_qual = arrival_pos_qual
    else:
        if arrival_pos_qual > arrival_neg_qual:
            arrival = arrival_pos
            arrival_qual = arrival_pos_qual
        else:
            arrival = arrival_neg
            arrival_qual = arrival_neg_qual
    
    if store_header != 'auto':
        phase_var = store_header
    
    arrival = shift_Max(seis, arrival)
    seis.stats.sac[phase_var] = arrival - shift
    seis.stats.sac['k'+phase_var] = phase_name+'ap'
    seis.stats.sac['user'+phase_var[-1]] = np.round(arrival_qual*100)
    seis.stats.sac['kuser0'] = 'PickQual'
    seis.write(file_dir + seis_file.rstrip('.s_fil') + '_auto' + '.sac')
    
    return

#if tf.test.is_gpu_available() == True:
#    pass
#else:
#    session = tf.Session(config = tf.ConfigProto())
#keras.losses.huber_loss = huber_loss
model_path = '../models/'
model = tf.keras.models.load_model(model_path+model_name)
#neg_model = load_model('../auto_seismo/models/arrival_SS_neg_model_0040.h5')
n_preds = time_window * resample_Hz # Maximum number of times the peak could be found, from sliding the window

files = np.sort([f for f in os.listdir(file_dir) if '.s_fil' in f])
gen_whitespace = lambda x: ' '*len(x)

print('\nPicking for', phase, 'phase in', len(files), 'files.')
for f, seis_file in enumerate(files):
    print_string = 'File ' + str(f+1) + ' / ' + str(len(files)) + '...'
    print('\r'+print_string, end=gen_whitespace(print_string))
    # Estimated picking time is 2 min (based on 1 sample)
    pick_Phase(file_dir, seis_file, phase, model)
print('\nSeismograms picked. Bon appetit!')