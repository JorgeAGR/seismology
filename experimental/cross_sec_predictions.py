#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 14:38:04 2020

@author: jorgeagr
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 17:06:47 2020

@author: jorgeagr
"""
import os
import argparse
import obspy
import numpy as np
import pandas as pd
#import tensorflow.keras.losses
#import tensorflow.keras.metrics
#from tensorflow.losses import huber_loss
from sklearn.cluster import DBSCAN
from sklearn.linear_model import LinearRegression
from scipy.interpolate import interp1d
from time import time as clock
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Settings for plots
golden_ratio = (np.sqrt(5) + 1) / 2
width = 25
height = width / golden_ratio

mpl.rcParams['figure.figsize'] = (width, height)
mpl.rcParams['font.size'] = 20
mpl.rcParams['figure.titlesize'] = 'large'
mpl.rcParams['legend.fontsize'] = 'medium'
mpl.rcParams['xtick.major.size'] = 12
mpl.rcParams['xtick.minor.size'] = 8
mpl.rcParams['xtick.labelsize'] = 18
mpl.rcParams['ytick.major.size'] = 12
mpl.rcParams['ytick.minor.size'] = 8
mpl.rcParams['ytick.labelsize'] = 18

def filter_Data(condition_array, *arrays):
    new_arrays = []
    for array in arrays:
        new_arrays.append(array[condition_array])
    return new_arrays

def get_Predictions(cap, qual_cut=0.6):
    df = pd.read_csv('/home/jorgeagr/Documents/seismology/experimental/cross_secs_dat/{}caps_deg_preds.csv'.format(cap))
    preds = 0 
    for key in df.keys():
        if 'pred' in key:
            preds += 1
    pred_inds = np.array([1+4*i for i in range(preds)])
    err_inds = pred_inds + 1
    amp_inds = err_inds + 1
    qual_inds = amp_inds + 1
    
    files = df['file'].values
    arrivals = df.values[:,pred_inds]#.flatten()
    errors = df.values[:,err_inds].flatten()
    amps = df.values[:,amp_inds].flatten()
    qualities = df.values[:,qual_inds]
    
    arrivals[qualities < qual_cut] = 0
    qualities = qualities.flatten()
    arrivals_inds = np.meshgrid(np.arange(arrivals.shape[1]),
                               np.arange(arrivals.shape[0]))[1].flatten()
    files = np.meshgrid(np.arange(arrivals.shape[1]), files)[1].flatten()
    arrivals = arrivals.flatten()
    
    arrivals_inds, qualities, errors, amps, arrivals, files = filter_Data(arrivals != 0, arrivals_inds,
                                                                           qualities, errors, amps, arrivals, files)
    arrivals_inds, qualities, errors, amps, arrivals, files = filter_Data(arrivals > -300, arrivals_inds,
                                                                           qualities, errors, amps, arrivals, files)
    
    return arrivals, errors, qualities, files

def discontinuity_model(dat_path, precursor, depth, model):
    df_disc = pd.read_csv('{}S{}Stimes_{}_{}.dat'.format(dat_path, precursor, depth, model), sep=' ', header=None)
    df_main = pd.read_csv('{}SStimes_{}_{}.dat'.format(dat_path, depth, model), sep=' ', header=None)
    df_disc[1] = (df_disc - df_main)[1]
    func = interp1d(df_disc[0].values, df_disc[1].values)
    
    return func

def find_Relevant(arrivals, errors, qualities, files, init_time, end_time, uncertainty, sigmas, weighted=True):
    condition = (arrivals < init_time + sigmas*uncertainty) & (arrivals > end_time - sigmas*uncertainty)
    arrivals = arrivals[condition]
    errors = errors[condition]
    qualities = qualities[condition]
    files = files[condition]
    '''
    percent_data = 1
    eps = 5
    clusters = []
    clustering_data = np.vstack([arrivals, gcarcs]).T
    while len(clusters) < 1:
        percent_data += -0.05
        dbscan = DBSCAN(eps=eps, min_samples=len(arrivals)*percent_data)#np.round(len(df)*percent_data))
        #dbscan.fit(arrivals.reshape(-1,1))
        dbscan.fit(clustering_data)
        clusters = np.unique(dbscan.labels_)
        if -1 in clusters:
            clusters = clusters[1:]
    
    linear = LinearRegression()
    linear.fit(gcarcs[dbscan.labels_==0].reshape(-1,1),
               arrivals[dbscan.labels_==0], sample_weight=(qualities[dbscan.labels_==0])**weighted)
    arrive_linear = linear.predict(gcarcs.reshape(-1,1)).flatten()
    
    zscore = (arrivals - arrive_linear) / uncertainty
    condition = np.abs(zscore) < sigmas
    gcarcs = gcarcs[condition]
    arrivals = arrivals[condition]
    errors = errors[condition]
    qualities = qualities[condition]
    files = files[condition]
    
    linear = LinearRegression()
    linear.fit(gcarcs.reshape(-1,1),
               arrivals, sample_weight=qualities**weighted)
    #return gcarcs_n, arrivals_n, errors_n, qualities_n, files_n, linear
    '''
    return arrivals, errors, qualities, files

def find_Relevant_OLD(arrivals, gcarcs, qualities, prem_model, iasp_model, uncertainty, sigmas):
    theory = (np.vstack([prem_model(gcarcs), iasp_model(gcarcs)]))
    inds = np.array([])
    for i in range(theory.shape[0]):
        zscore = (arrivals - theory[i, :]) / uncertainty
        inds = np.concatenate([inds, np.argwhere(np.abs(zscore) < sigmas).flatten()])
    #zscore = (arrivals - theory) / uncertainty
    condition = np.asarray(np.unique(inds), dtype=np.int)#np.abs(zscore) < sigmas
    gcarcs = gcarcs[condition]
    arrivals = arrivals[condition]
    qualities = qualities[condition]
    return gcarcs, arrivals, qualities, files
    
q1 = 0.6
#q2 = 0.7

cap = 5
arrivals, errors, qualities, files = get_Predictions(cap, qual_cut=q1)

x = np.linspace(100, 180, num=100)
# 410 models
prem_410_0 = discontinuity_model('/home/jorgeagr/Documents/seismology/experimental/ss_ind_precursors/phase_times/',
                                '410', '0', 'prem')
iasp_410_70 = discontinuity_model('/home/jorgeagr/Documents/seismology/experimental/ss_ind_precursors/phase_times/',
                                '410', '70', 'iasp')
# 660 models
prem_660_0 = discontinuity_model('/home/jorgeagr/Documents/seismology/experimental/ss_ind_precursors/phase_times/',
                                '660', '0', 'prem')
prem_660_70 = discontinuity_model('/home/jorgeagr/Documents/seismology/experimental/ss_ind_precursors/phase_times/',
                                '660', '70', 'prem')
iasp_660_0 = discontinuity_model('/home/jorgeagr/Documents/seismology/experimental/ss_ind_precursors/phase_times/',
                                '660', '0', 'iasp')
iasp_660_70 = discontinuity_model('/home/jorgeagr/Documents/seismology/experimental/ss_ind_precursors/phase_times/',
                                '660', '70', 'iasp')

prem_520_0 = discontinuity_model('/home/jorgeagr/Documents/seismology/experimental/ss_ind_precursors/phase_times/',
                                '520', '0', 'prem')
iasp_520_0 = discontinuity_model('/home/jorgeagr/Documents/seismology/experimental/ss_ind_precursors/phase_times/',
                                '520', '0', 'iasp')

min410 = np.min(np.hstack([prem_410_0(x), iasp_410_70(x)]))
max410 = np.max(np.hstack([prem_410_0(x), iasp_410_70(x)]))
min660 = np.min(np.hstack([prem_660_70(x), iasp_660_0(x)]))
max660 = np.max(np.hstack([prem_660_70(x), iasp_660_0(x)]))

# Weighted
arrivals410, errors410, qualities410, files410 = find_Relevant(arrivals, errors, qualities, files,
                                                                          max410, min410, 2.5, 2) 
arrivals660, errors660, qualities660, files660 = find_Relevant(arrivals, errors, qualities, files,
                                                                          max660, min660, 2.5, 2)

df410 = pd.DataFrame(data={'file':files410,
                           'time': arrivals410,
                           'error': errors410,
                           'quality': qualities410})
df410.to_csv('ss_ind_precursors/S410S_{}cap_results.csv'.format(cap), index=False)
df660 = pd.DataFrame(data={'file': files660,
                           'time': arrivals660,
                           'error': errors660,
                           'quality': qualities660})
df660.to_csv('ss_ind_precursors/S660S_{}cap_results.csv'.format(cap), index=False)
