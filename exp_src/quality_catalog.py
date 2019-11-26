# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 00:17:45 2019

@author: jorge
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
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

height = 8#16
width = 12#25

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

def get_Lauren_Pred(cap, precursor, file):
    file_path = 'cross_secs_dat/lauren_pred/' + cap + 'caps_deg.dat'
    if precursor == '410':
        ind = 1
    elif precursor == '660':
        ind = 4
    df = pd.read_csv(file_path, header=None, sep=' ')
    dat_bins = df[0].values
    file_ind = np.where(dat_bins == file)[0][0]
    dat_times = np.asarray([line.split('+/-')[0] for line in df[ind].values], dtype=np.float)
    dat_errors = np.asarray([line.split('+/-')[1] for line in df[ind].values], dtype=np.float)
    return dat_times[file_ind], dat_errors[file_ind]

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

cap = '5'

pred_file = pd.read_csv('cross_secs_dat/model_pred/'+cap+'caps_deg_results.csv')

# By indexing with [similar_inds][similar_sort], this returns a subset of
# quality factors sorted from ratios 0.9 to 1.2, index 212 is the first ratio=1
# Try to take a sample in increments of 0.1 with as close a ratio to 1
qual_similarity = pred_file['410qual'].values / pred_file['660qual'].values
similar_inds = np.where((qual_similarity > 0.9) & (qual_similarity < 1.1))
similar_sort = np.argsort(qual_similarity[similar_inds])
qualities = pred_file[['410qual', '660qual']].values[similar_inds][similar_sort]
files = pred_file['file'].values[similar_inds][similar_sort]
arrivals410 = pred_file['410pred'].values[similar_inds][similar_sort]
arrivals660 = pred_file['660pred'].values[similar_inds][similar_sort]

# Specific for the SS cross-secs 5cap bins
qual_dic = {'0.1': [173,], '0.2': [219,], '0.3': [81,],'0.4': [233, 232, 185],
            '0.5': [264, 216, 226], '0.6': [225, 203, 194], '0.7': [224, 256, 201],
            '0.8': [191, 181, 254],'0.9': [209, 214, 196],'1.0': [283,]}

for i, key in enumerate(qual_dic):
    index = qual_dic[key][0]
    file = files[index]
    arr_410 = arrivals410[index]
    arr_660 = arrivals660[index]
    qual_410 = qualities[index,0]
    qual_660 = qualities[index,1]
    quals = [qual_660, qual_410]
    
    try:
        lauren_410, lerr_410 = get_Lauren_Pred(cap, '410', file)
        lauren_660, lerr_660 = get_Lauren_Pred(cap, '660', file)
        l_errors = [lerr_660, lerr_410]
    except:
        pass
    
    cs = obspy.read('../../seismograms/cross_secs/' + cap + 'caps_deg/' + file + '.sac')
    cs = cs[0].resample(10)
    
    shift = -cs.stats.sac.b
    times = cs.times() - shift
    b = cs.stats.sac.b + shift
    e = cs.stats.sac.e + shift
    
    cs_norm = cs.data / np.abs(cs.data).max()
    fig, ax = plt.subplots()
    ax.plot(times, cs_norm, color='black')
    for i, ar in enumerate([arr_660, arr_410]):
        print(ar)
        ax.axvline(ar, color='blue', linestyle='--')
        ax.text(ar-15, 0.2, quals[i], rotation=90, fontsize=16)
    ax.axvline(ar, color='blue', linestyle='--', label='model')
    try:
        for i, ar in enumerate([lauren_660, lauren_410]):
            ax.axvline(ar, color='red', linestyle=':')
        ax.axvline(ar, color='red', linestyle=':', label='bootstrap')
    except:
        pass
    #for i, ar in enumerate(arrivals_neg[np.argsort(counts_neg)][-5:]):
    #   ax.axvline(ar, color='red', linestyle='--')
    #   ax.text(ar-5, 0.1, np.sort(counts_neg)[-5:][i], rotation=90, fontsize=16)
    #   ax.axvline(ar, color='red', linestyle='--', label='negative model')
    ax.set_ylim(cs_norm.min(), cs_norm.max())
    ax.set_xlim(times.min(), times.max())
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Amplitude')
    ax.set_title(cap + 'cap/' + file)
    ax.xaxis.set_major_locator(mtick.MultipleLocator(50))
    ax.xaxis.set_minor_locator(mtick.MultipleLocator(25))
    ax.yaxis.set_major_locator(mtick.MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(mtick.MultipleLocator(0.25))
    ax.legend(loc='upper left')
    fig.tight_layout(pad=0.5)
    fig.savefig('../figs/cross_secs/quality_eg/qual' + key.replace('.','') + '.eps', dpi=500)
    plt.close()