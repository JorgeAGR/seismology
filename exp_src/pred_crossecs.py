#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 10:34:06 2019

@author: jorgeagr
"""

import os
import argparse
import obspy
import numpy as np
import pandas as pd
from keras.models import load_model
import keras.losses
import keras.metrics
from tensorflow.losses import huber_loss
from sklearn.cluster import DBSCAN

resample_Hz = 10
time_window = 40

# Sort-of global vars. Optional inputs for user
relevant_preds = 5
# For clustering to find 410 and 660
eps=5
percent_data=0.99

# User inputs
cap_size = 15
file_dir = '../../seismograms/cross_secs/' + str(cap_size) + 'caps_deg/'
find660 = True
model_path = '../auto_seismo/models/arrival_SS_pos_model_0040.h5'
write_path = './cross_secs_dat/'

parser = argparse.ArgumentParser(description='Predict precursor arrivals in vespagram cross-sectional data.')
parser.add_argument('file_dir', help='Cross-section SAC files directory.', type=str, default=file_dir)
parser.add_argument('find660', help='Require the 660 discontinuity to be found.', type=bool, default=find660)
parser.add_argument('model_path', help='Path to model H5 file.', type=str, default=model_path)
parser.add_argument('write_path', help='Path to write predicition result CSVs.', type=str, default=write_path)
parser.add_argument('-N', metavar='relevant_preds', help='Number of top predictions to consider.',
                    type=int, default=relevant_preds)
parser.add_argument('-e', metavar='eps', help='Clustering maximum distance from core point.', type=int, default=eps)
parser.add_argument('-p', metavar='percent_data', help='Percentage of data to define a core point.',
                    type=float, default=percent_data)
args = parser.parse_args()

file_dir = args.file_dir
find660 = args.find660
model_path = args.model_path
write_path = args.write_path
relevant_preds = args.N
eps = args.e
percent_data = args.p

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

def find_Precursors(file_dir, cs_file, model, relevant_preds=5):
    cs = obspy.read(file_dir+cs_file)
    cs_file = cs_file.rstrip('.sac')
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
    #relevant_preds = 5
    discont_ind = np.argsort(counts)[-relevant_preds:]
    clusters = clusters[discont_ind]
    counts = counts[discont_ind]
    arrivals_pos = np.zeros(relevant_preds)
    arrivals_pos_err = np.zeros(relevant_preds)
    arrivals_amps = np.zeros(relevant_preds)
    arrivals_quality = np.zeros(relevant_preds)
    for i, c in enumerate(clusters):
        arrivals_pos[i] = np.mean(window_preds[dbscan.labels_ == c])
        arrivals_pos_err[i] = np.std(window_preds[dbscan.labels_ == c])
        arrivals_quality[i] = counts[i] / n_preds
        initamp = np.where(times < arrivals_pos[i])[0][-1]
        arrivals_amps[i] = cs.data[initamp:initamp+2].max()
    arrivals_pos = arrivals_pos - shift
    '''
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
    '''
    make_string = lambda x: str(arrivals_pos[x]) + ',' + str(arrivals_pos_err[x]) + ',' + str(arrivals_amps[x]) + ',' + str(arrivals_quality[x])
    result_strings = [make_string(i) for i in range(relevant_preds)]
    return result_strings

def find_410_660(pred_csv_path, eps=eps, percent_data=percent_data):
    df = pd.read_csv(pred_csv_path)
    pred_inds = np.asarray([1, 5, 9, 13, 17])
    err_inds = pred_inds + 1
    amp_inds = err_inds + 1
    qual_inds = amp_inds + 1
    
    arrivals = df.values[:,pred_inds].flatten()
    errors = df.values[:,err_inds]
    amps = df.values[:,amp_inds]
    qualities = df.values[:,qual_inds]
    
    # Parameters to play with: eps, min_samples
    dbscan = DBSCAN(eps=eps, min_samples=np.round(len(df)*percent_data))
    dbscan.fit(arrivals.reshape(-1,1))
    
    clusters = np.unique(dbscan.labels_)
    if -1 in clusters:
        clusters = clusters[1:]
    # This part assumes there are only 2 labels, one for each global discontinuity
    # May have to modify clustering part in case less than or more than 2 appear?
    # Purpose is to find corresponding label for each discontinuity
    avg_preds = np.asarray([arrivals[dbscan.labels_ == c].mean() for c in clusters])
    sort_ind = np.argsort(avg_preds)
    cluster660, cluster410 = clusters[sort_ind]
    
    # Reshape into shape of arrivals, to see label for each predicted arrival
    # of each bin
    labels = dbscan.labels_.reshape(len(df), len(pred_inds))
    
    ind410 = np.zeros(len(df), dtype=np.int)
    ind660 = np.zeros(len(df), dtype=np.int)
    
    for i in range(len(labels)):
        l, counts = np.unique(labels[i], return_counts=True)
        # If either discontinuity isn't found, label to trash later
        if (0 not in l) or (1 not in l):
            labels[i] = -np.ones(len(labels[i]))
            ind410[i] = -1
            ind660[i] = -1
            continue
        for c in clusters:
            if counts[l==c][0] > 1:
                where = np.argwhere(labels[i]==c).flatten()
                maxqual = np.argmax(qualities[i][where])
                throw = [j for j in range(len(where)) if j != maxqual]
                labels[i][where[throw]] = -1
        ind410[i] = np.where(labels[i]==cluster410)[0][0]
        ind660[i] = np.where(labels[i]==cluster660)[0][0]
        
    arrivals = arrivals.reshape(len(df), len(pred_inds))
    
    found410 = np.where(ind410 != -1)[0]
    found660 = np.where(ind660 != -1)[0]
    foundboth = np.union1d(found410, found660)
    ind410 = ind410[foundboth]
    ind660 = ind660[foundboth]
    
    preds410, preds660 = arrivals[foundboth,ind410], arrivals[foundboth,ind660]
    errs410, errs660 = errors[foundboth,ind410], errors[foundboth,ind660]
    amps410, amps660 = amps[foundboth,ind410], amps[foundboth,ind660]
    quals410,quals660 = qualities[foundboth,ind410], qualities[foundboth,ind660]
    
    df_disc = pd.DataFrame(data = {'file':df['file'].values[foundboth],
                                  '410pred':preds410, '410err':errs410, '410amp':amps410, '410qual':quals410,
                                  '660pred':preds660, '660err':errs660, '660amp':amps660, '660qual':quals660})
    return df_disc

def find_410(pred_csv_path, eps=eps, percent_data=percent_data):
    df = pd.read_csv(pred_csv_path)
    pred_inds = np.asarray([1, 5, 9, 13, 17])
    err_inds = pred_inds + 1
    amp_inds = err_inds + 1
    qual_inds = amp_inds + 1
    
    arrivals = df.values[:,pred_inds].flatten()
    errors = df.values[:,err_inds]
    amps = df.values[:,amp_inds]
    qualities = df.values[:,qual_inds]
    
    # Parameters to play with: eps, min_samples
    dbscan = DBSCAN(eps=eps, min_samples=np.round(len(df)*percent_data))
    dbscan.fit(arrivals.reshape(-1,1))
    
    clusters = np.unique(dbscan.labels_)
    if -1 in clusters:
        clusters = clusters[1:]
    # This part assumes there are only 2 labels, one for each global discontinuity
    # May have to modify clustering part in case less than or more than 2 appear?
    # Purpose is to find corresponding label for each discontinuity
    avg_preds = np.asarray([arrivals[dbscan.labels_ == c].mean() for c in clusters])
    sort_ind = np.argsort(avg_preds)
    cluster660, cluster410 = clusters[sort_ind]
    
    # Reshape into shape of arrivals, to see label for each predicted arrival
    # of each bin
    labels = dbscan.labels_.reshape(len(df), len(pred_inds))
    
    ind410 = np.zeros(len(df), dtype=np.int)
    ind660 = np.zeros(len(df), dtype=np.int)
    
    for i in range(len(labels)):
        l, counts = np.unique(labels[i], return_counts=True)
        # If either discontinuity isn't found, label to trash later
        if (0 not in l) or (1 not in l):
            labels[i] = -np.ones(len(labels[i]))
            ind410[i] = -1
            ind660[i] = -1
            continue
        for c in clusters:
            if counts[l==c][0] > 1:
                where = np.argwhere(labels[i]==c).flatten()
                maxqual = np.argmax(qualities[i][where])
                throw = [j for j in range(len(where)) if j != maxqual]
                labels[i][where[throw]] = -1
        ind410[i] = np.where(labels[i]==cluster410)[0][0]
        ind660[i] = np.where(labels[i]==cluster660)[0][0]
        
    arrivals = arrivals.reshape(len(df), len(pred_inds))
    
    found410 = np.where(ind410 != -1)[0]
    found660 = np.where(ind660 != -1)[0]
    foundboth = np.union1d(found410, found660)
    ind410 = ind410[foundboth]
    ind660 = ind660[foundboth]
    
    preds410, preds660 = arrivals[foundboth,ind410], arrivals[foundboth,ind660]
    errs410, errs660 = errors[foundboth,ind410], errors[foundboth,ind660]
    amps410, amps660 = amps[foundboth,ind410], amps[foundboth,ind660]
    quals410,quals660 = qualities[foundboth,ind410], qualities[foundboth,ind660]
    
    df_disc = pd.DataFrame(data = {'file':df['file'].values[foundboth],
                                  '410pred':preds410, '410err':errs410, '410amp':amps410, '410qual':quals410,
                                  '660pred':preds660, '660err':errs660, '660amp':amps660, '660qual':quals660})
    return df_disc

keras.losses.huber_loss = huber_loss
pos_model = load_model(model_path)
#neg_model = load_model('../auto_seismo/models/arrival_SS_neg_model_0040.h5')
n_preds = time_window * resample_Hz # Maximum number of times the peak could be found, from sliding the window

files = np.sort([f.rstrip('.sac') for f in os.listdir(file_dir) if '.sac' in f])

with open(write_path + file_dir.split('/')[-2] + '_preds.csv', 'w+') as pred_csv:
    '''
    print('file,410pred,410err,410amp,410qual,660pred,660err,660amp,660qual', file=pred_csv)
    '''
    print_cols = lambda x: 'pred{0},err{0},amp{0},qual{0},'.format(x)
    header_string = 'file,'
    for i in range(relevant_preds):
        header_string += print_cols(i)
    header_string = header_string.rstrip(',')
    print(header_string, file=pred_csv)

for f, cs_file in enumerate(files):
    '''
    print('File', f+1, '/', len(files),'...', end=' ')
    string_410, string_660 = find_Precursors(file_dir, cs_file+'.sac', pos_model)
    with open(file_dir.split('/')[-2] + '_preds.csv', 'a') as pred_csv:
        print(cs_file + ',' + string_410 + ',' + string_660, file=pred_csv)
    '''
    print('File', f+1, '/', len(files),'...', end=' ')
    # Removed and readded the .sac extension due to getting different sorting
    # of files when leaving the extension in the string
    results = find_Precursors(file_dir, cs_file+'.sac', pos_model, relevant_preds)
    with open(write_path + file_dir.split('/')[-2] + '_preds.csv', 'a') as pred_csv:
        print(cs_file, end=',', file=pred_csv)
        for i in range(relevant_preds-1):
            print(results[i], end=',', file=pred_csv)
        print(results[-1], file=pred_csv)

if find660:
    found = find_410_660(write_path + file_dir.split('/')[-2] + '_preds.csv')
else:
    found = find_410(write_path + file_dir.split('/')[-2] + '_preds.csv')
found.to_csv(write_path + file_dir.split('/')[-2] + '_results.csv', index=False)