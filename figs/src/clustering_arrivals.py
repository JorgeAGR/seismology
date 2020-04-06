#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 15:20:25 2020

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

from sklearn.cluster import Birch

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

def get_Predictions(pred_csv_path, file, qual_cut=0.6):
    df = pd.read_csv(pred_csv_path + file)
    preds = 0 
    for key in df.keys():
        if 'pred' in key:
            preds += 1
    pred_inds = np.array([2+4*i for i in range(preds)])
    err_inds = pred_inds + 1
    amp_inds = err_inds + 1
    qual_inds = amp_inds + 1
    
    files = df['file'].values
    arrivals = df.values[:,pred_inds]#.flatten()
    gcarcs = df['gcarc'].values
    errors = df.values[:,err_inds].flatten()
    amps = df.values[:,amp_inds].flatten()
    qualities = df.values[:,qual_inds]
    
    arrivals[qualities < qual_cut] = 0
    qualities = qualities.flatten()
    arrivals_inds = np.meshgrid(np.arange(arrivals.shape[1]),
                               np.arange(arrivals.shape[0]))[1].flatten()
    gcarcs = np.meshgrid(np.arange(arrivals.shape[1]), gcarcs)[1].flatten()
    arrivals = arrivals.flatten()
    
    arrivals_inds, gcarcs, qualities, errors, amps, arrivals = filter_Data(arrivals != 0, arrivals_inds, gcarcs,
                                                                           qualities, errors, amps, arrivals)
    arrivals_inds, gcarcs, qualities, errors, amps, arrivals = filter_Data(gcarcs >= 100, arrivals_inds, gcarcs,
                                                                           qualities, errors, amps, arrivals)
    arrivals_inds, gcarcs, qualities, errors, amps, arrivals = filter_Data(arrivals > -300, arrivals_inds, gcarcs,
                                                                           qualities, errors, amps, arrivals)
    
    return gcarcs, arrivals, qualities

    
q1 = 0.6
init_time = -130
end_time = -185
weighted = True
uncertainty = 5
sigmas = 2
#q2 = 0.7
gcarcs, arrivals, qualities = get_Predictions('/home/jorgeagr/Documents/seismology/experimental/ss_ind_precursors/',
                            'SS_kept_preds.csv', qual_cut=q1)

condition = (arrivals < init_time) & (arrivals > end_time)
arrivals = arrivals[condition]
gcarcs = gcarcs[condition]
qualities = qualities[condition]
clustering_data = np.vstack([arrivals, gcarcs]).T

percent_data = 1
eps = 5
clusters = []
while len(clusters) < 1:
    percent_data += -0.05
    dbscan = DBSCAN(eps=eps, min_samples=len(arrivals)*percent_data)#np.round(len(df)*percent_data))
    #dbscan.fit(arrivals.reshape(-1,1))
    dbscan.fit(clustering_data)
    clusters = np.unique(dbscan.labels_)
    if -1 in clusters:
        clusters = clusters[1:]

# This groups it much better, but idk how general it is.
#dbscan = DBSCAN(eps=2.5, min_samples=100)
#dbscan.fit(clustering_data)

linear = LinearRegression()
linear.fit(gcarcs[dbscan.labels_==0].reshape(-1,1),
           arrivals[dbscan.labels_==0], sample_weight=(qualities[dbscan.labels_==0])**weighted)
arrive_linear = linear.predict(gcarcs.reshape(-1,1)).flatten()

x = np.linspace(100, 180, num=100)
y = linear.predict(x.reshape(-1,1))
fig, ax = plt.subplots()
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size="20%", pad=0.05)

ax.set_xlim(90, 180)
ax.set_ylim(-100, -300)
ax.yaxis.set_major_locator(mtick.MultipleLocator(50))
ax.yaxis.set_minor_locator(mtick.MultipleLocator(10))
ax.xaxis.set_major_locator(mtick.MultipleLocator(20))
ax.xaxis.set_minor_locator(mtick.MultipleLocator(10))
ax.set_ylabel('SdS-SS (s)')
ax.set_xlabel('Epicentral Distance (deg)')
ax.invert_yaxis()
ax.fill_between(x, y-uncertainty*sigmas, y+uncertainty*sigmas, color='lightblue', label='Considered')
ax.scatter(gcarcs, arrivals, marker='.', color='black')
ax.scatter(gcarcs[dbscan.labels_==0], arrivals[dbscan.labels_==0], marker='.', color='blue', label='Clustered')
ax.plot(x, linear.predict(x.reshape(-1,1)).flatten(), color='red', linewidth=2, label='Initial Fit')

cax.set_xlim(120, 130)
cax.set_ylim(-150, -160)
cax.invert_yaxis()
cax.yaxis.tick_right()
cax.xaxis.tick_top()
cax.yaxis.set_major_locator(mtick.MultipleLocator(5))
cax.yaxis.set_minor_locator(mtick.MultipleLocator(1))
cax.xaxis.set_major_locator(mtick.MultipleLocator(10))
cax.xaxis.set_minor_locator(mtick.MultipleLocator(2))
cax.scatter(gcarcs, arrivals, marker='.', color='black')
cax.scatter(gcarcs[dbscan.labels_==0], arrivals[dbscan.labels_==0], marker='.', color='blue')
cax.plot(x, y.flatten(), color='red', linewidth=2)

zscore = (arrivals - arrive_linear) / uncertainty
condition = np.abs(zscore) < sigmas
gcarcs = gcarcs[condition]
arrivals = arrivals[condition]
qualities = qualities[condition]

linear = LinearRegression()
linear.fit(gcarcs.reshape(-1,1),
           arrivals, sample_weight=qualities**weighted)
ax.plot(x, linear.predict(x.reshape(-1,1)).flatten(), color='orange', linewidth=2, label='Final Fit')
cax.plot(x, linear.predict(x.reshape(-1,1)).flatten(), color='orange', linewidth=2)
ax.legend()
fig.tight_layout(pad=0.5)
fig.savefig('../clustering_arrivals.png', dpi=200)
