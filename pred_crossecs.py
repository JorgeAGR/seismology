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
from tensorflow.keras.models import load_model
#import tensorflow.keras.losses
#import tensorflow.keras.metrics
#from tensorflow.losses import huber_loss
from sklearn.cluster import DBSCAN
from time import time as clock

parser = argparse.ArgumentParser(description='Predict precursor arrivals in vespagram cross-sectional data.')
parser.add_argument('file_dir', help='Cross-section SAC files directory.', type=str)#, default=file_dir)
#parser.add_argument('find660', help='Require the 660 discontinuity to be found.', type=bool)
parser.add_argument('write_path', help='Path to write predicition result CSVs.', type=str)
parser.add_argument('model_path', help='Path to model H5 file.', type=str)
parser.add_argument('-N', metavar='relevant_preds', help='Number of top predictions to consider.',
                    type=int, default=5)
parser.add_argument('-eps', metavar='eps', help='Clustering maximum distance from core point.', type=int, default=5)
parser.add_argument('-p', metavar='percent_data', help='Percentage of data to define a core point.',
                    type=float, default=1)
parser.add_argument('-b', metavar='begin_pred', help='Seconds before main arrival to begin precursor search.',
                    type=float, default=-400)
parser.add_argument('-e', metavar='end_pred', help='Seconds before main arrival to end precursor search',
                    type=float, default=-80)
parser.add_argument('-n660', help='Ignore the 660 discontinuity from being found.', action='store_false')
args = parser.parse_args()

file_dir = args.file_dir
find660 = args.n660
model_path = args.model_path
write_path = args.write_path
relevant_preds = args.N
eps = args.eps
percent_data = args.p
begin_pred = args.b
end_pred = args.e

check_Path_String = lambda x: x+'/' if x[-1] != '/' else x
file_dir = check_Path_String(file_dir)
write_path = check_Path_String(write_path)
begin_pred = -np.abs(begin_pred) # Seconds before main arrival.
end_pred = -np.abs(end_pred) # ditto above

def cut_Window(cross_sec, times, t_i, t_f):
    #init = np.where(times == np.round(t_i, 1))[0][0]
    #end = np.where(times == np.round(t_f, 1))[0][0]
    init = int(np.round(t_i*resample_Hz))
    end = int(np.round(t_f*resample_Hz))
    
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

def scan(seis, times, time_i_grid, time_f_grid, shift, model, negative=False):
    window_preds = np.zeros(len(time_i_grid))
    for i, t_i, t_f in zip(range(len(time_i_grid)), time_i_grid, time_f_grid):
        seis_window = cut_Window(seis, times, t_i, t_f) * (-1)**negative
        seis_window = seis_window / np.abs(seis_window).max()
        # Take the absolute value of the prediction to remove any wonky behavior in finding the max
        # Doesn't matter since they are bad predictions anyways
        window_preds[i] += np.abs(model.predict(seis_window.reshape(1, len(seis_window), 1))[0][0]) + t_i
    return window_preds

def find_Precursors(file_dir, sac_file, model, relevant_preds, pred_init_t, pred_end_t):
    cs = obspy.read(file_dir+sac_file+extension)
    sac_file = sac_file.rstrip(extension)
    cs = cs[0].resample(resample_Hz)
    times = cs.times()
    shift = -cs.stats.sac.b
    
    begin_time = np.round(pred_init_t + shift, decimals=1)
    end_time = np.round(pred_end_t + shift, decimals=1)
    
    time_i_grid = np.arange(begin_time, end_time - time_window + 0.1, 0.1)
    time_f_grid = np.arange(begin_time + time_window, end_time + 0.1, 0.1)
    window_preds = np.zeros(len(time_i_grid))
    window_preds = scan(cs, times, time_i_grid, time_f_grid, shift, model)
    
    #print('Finding arrivals...', end=' ')
    dbscan = DBSCAN(eps=0.05, min_samples=2)
    dbscan.fit(window_preds.reshape(-1,1))
    clusters, counts = np.unique(dbscan.labels_, return_counts=True)
    if -1 in clusters:
        clusters = clusters[1:]
        counts = counts[1:]
    #relevant_preds = 5e_string(i) for i in range(relevant_preds)]
    discont_ind = np.argsort(counts)[-relevant_preds:]
    clusters = clusters[discont_ind]
    counts = counts[discont_ind]
    arrivals = np.zeros(relevant_preds)
    arrivals_err = np.zeros(relevant_preds)
    arrivals_amps = np.zeros(relevant_preds)
    arrivals_quality = np.zeros(relevant_preds)
    for i, c in enumerate(clusters):
        arrivals[i] = np.mean(window_preds[dbscan.labels_ == c])
        arrivals_err[i] = np.std(window_preds[dbscan.labels_ == c])
        arrivals_quality[i] = counts[i] / n_preds
        initamp = np.where(times < arrivals[i])[0][-1]
        arrivals_amps[i] = cs.data[initamp:initamp+2].max()
    arrivals = arrivals - shift
    
    make_string = lambda x: '{},{},{},{}'.format(arrivals[x],arrivals_err[x],arrivals_amps[x],arrivals_quality[x])
    result_strings = list(map(make_string, range(relevant_preds)))
    return cs.stats.sac.gcarc, result_strings

def prepare_Pred_CSV(discont, discont_ind_list, files, arrivals,
                     arrivals_inds, qualities, errors, amps):
    arrivals = arrivals[discont_ind_list]
    arrivals_inds = arrivals_inds[discont_ind_list]
    qualities = qualities[discont_ind_list]
    errors = errors[discont_ind_list]
    amps = amps[discont_ind_list]
    files = files[arrivals_inds]
    '''
    add GCARC? probably not, only need it to figure out good preds.
    '''
    
    df = pd.DataFrame(data={'file': files,
                            '{}pred'.format(discont): arrivals,
                            '{}err'.format(discont): errors,
                            '{}amp'.format(discont): amps,
                            '{}qual'.format(discont): qualities})
    return df

def find_410_660(pred_csv_path, qual_cut=0.6, eps=eps, percent_data=percent_data):
    # Might have to update this whole code? If I consider the quality cutoff in the
    # individual scan instead. Might have to even write from scratch in that case.
    # Could still be worth it tho. - 03/03
    df = pd.read_csv(pred_csv_path)
    pred_inds = np.asarray([2, 6, 10, 14, 18])
    err_inds = pred_inds + 1
    amp_inds = err_inds + 1
    qual_inds = amp_inds + 1
    
    files = df['file'].values
    arrivals = df.values[:,pred_inds]#.flatten()
    gcarcs = df['gcarc'].values
    errors = df.values[:,err_inds]
    amps = df.values[:,amp_inds]
    qualities = df.values[:,qual_inds]
    
    del df
    
    arrivals[qualities < qual_cut] = 0
    arrivals_inds = np.meshgrid(np.arange(arrivals.shape[1]),
                               np.arange(arrivals.shape[0]))[1]
    gcarcs = np.meshgrid(np.arange(arrivals.shape[1]), gcarcs)[1]
    arrivals = arrivals.flatten()
    
    arrivals_inds = arrivals_inds.flatten()[arrivals != 0]
    gcarcs = gcarcs.flatten()[arrivals != 0]
    qualities = qualities.flatten()[arrivals != 0]
    errors = errors.flatten()[arrivals != 0]
    amps = amps.flatten()[arrivals != 0]
    arrivals = arrivals[arrivals != 0]
    
    '''
    GCARC is needed. Fix from this point onwards to include it.
    '''
    return 0
    # Parameters to play with: eps, min_samples
    percent_data = 0.95
    eps = 5
    clusters = []
    clustering_data = np.vstack([arrivals, gcarcs]).T
    while len(clusters) < 2:
        dbscan = DBSCAN(eps=eps, min_samples=len(arrivals)*percent_data)#np.round(len(df)*percent_data))
        #dbscan.fit(arrivals.reshape(-1,1))
        dbscan.fit(clustering_data)
        
        clusters = np.unique(dbscan.labels_)
        if -1 in clusters:
            clusters = clusters[1:]
        percent_data += -0.05
    
    # This part assumes there are only 2 labels, one for each global discontinuity
    # May have to modify clustering part in case less than or more than 2 appear?
    # Purpose is to find corresponding label for each discontinuity
    avg_preds = np.asarray([arrivals[dbscan.labels_ == c].mean() for c in clusters])
    cluster660, cluster410 = clusters[np.argsort(avg_preds)]
    # number of records where each discont found
    # arrivals[dbscan.labels_ == 0].shape[0]
    
    # Either I incldue all points in the clsuter, or just the core points.
    # Or perhaps a time cutoff.
    #index_method = dbscan.core_sample_indices_
    index_method = np.isin(dbscan.labels_, clusters)
    
    arrivals = arrivals[index_method]
    arrivals_inds = arrivals_inds[index_method]
    errors = errors[index_method]
    amps = amps[index_method]
    qualities = qualities[index_method]
    
    ind_list_410 = dbscan.labels_[index_method] == cluster410
    ind_list_660 = dbscan.labels_[index_method] == cluster660
    
    df410 = prepare_Pred_CSV('410', ind_list_410, files,
                             arrivals, arrivals_inds, qualities, errors, amps)
    df660 = prepare_Pred_CSV('660', ind_list_660, files,
                             arrivals, arrivals_inds, qualities, errors, amps)
    
    '''
    #I think anything below this is incompatible with the new version above
    
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
    '''
    return df410, df660

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

resample_Hz = 10
time_window = 40

pred_model = load_model(model_path)
n_preds = time_window * resample_Hz # Maximum number of times the peak could be found, from sliding the window

files = np.sort(os.listdir(file_dir))
extension = '.{}'.format(files[0].split('.')[-1])
files = np.sort([f.rstrip(extension) for f in files if extension in f])
gen_whitespace = lambda x: ' '*len(x)
pred_time = 0

with open(write_path + file_dir.split('/')[-2] + '_preds.csv', 'w+') as pred_csv:
    print_cols = lambda x: 'pred{0},err{0},amp{0},qual{0},'.format(x)
    header_string = 'file,gcarc,'
    for i in range(relevant_preds):
        header_string += print_cols(i)
    header_string = header_string.rstrip(',')
    print(header_string, file=pred_csv)

for f, sac_file in enumerate(files):
    #print('File', f+1, '/', len(files),'...', end=' ')
    print_string = 'File {} / {}... Est. Time per Prediction: {:.2f} sec'.format(f+1, len(files), pred_time)
    print('\r'+print_string, end=gen_whitespace(print_string))
    tick = clock()
    gcarc, results = find_Precursors(file_dir, sac_file+extension, pred_model, 
                              relevant_preds, begin_pred, end_pred)
    tock = clock()
    if f == 0:
        pred_time = tock-tick
    with open(write_path + file_dir.split('/')[-2] + '_preds.csv', 'a') as pred_csv:
        print('{},{}'.format(sac_file, gcarc), end=',', file=pred_csv)
        for i in range(relevant_preds-1):
            print(results[i], end=',', file=pred_csv)
        print(results[-1], file=pred_csv)
'''
# need to incorporate file mergning for multijob approaches before this line
df410, df660 = find_410_660(write_path + file_dir.split('/')[-2] + '_preds.csv')
df410.to_csv(write_path + file_dir.split('/')[-2] + '_results.csv', index=False)

# outdated with new version
if find660:
    found = find_410_660(write_path + file_dir.split('/')[-2] + '_preds.csv')
else:
    found = find_410(write_path + file_dir.split('/')[-2] + '_preds.csv')
found.to_csv(write_path + file_dir.split('/')[-2] + '_results.csv', index=False)
'''