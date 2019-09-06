#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 13:43:16 2019

@author: jorgeagr
"""
import os
import obspy
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


golden_ratio = (np.sqrt(5) + 1) / 2
width = 12
height = width / golden_ratio

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

# Change to get actuals for files in shifting_npz
def test_pred(datadir):
    files = np.sort(os.listdir(datadir))
    actuals = []
    for file in os.listdir(datadir):
        seismogram = obspy.read(datadir + file)[0]
        actuals.append(seismogram.stats.sac.t6 - seismogram.stats.sac.b)
    actuals = np.asarray(actuals)
    return files, actuals

shifting_npz = np.load('../train_data_shift_pred.npz')
files = shifting_npz['files'].astype(np.str)
pred_avg = shifting_npz['pred_avg']
pred_err = shifting_npz['pred_err']
flipped = shifting_npz['flipped']

actuals_npz = np.load('../train_data_simple_pred.npy')
actuals_files = actuals_npz['files'].astype(np.str)
arrivals = actuals_npz['arrivals']
'''
avail_files = []
avail_arrivals = []
for i, file in enumerate(actuals_files):
    if str(file.rstrip('.s_fil')) in files:
        avail_files.append(file)
        avail_arrivals.append(arrivals[i])
dir_files = np.asarray(avail_files)
arrivals = np.asarray(avail_arrivals)

shift_error = pred_avg - arrivals

fig, ax = plt.subplots()
weights = np.ones_like(shift_error)/len(shift_error)
hist = ax.hist(shift_error, np.arange(-1, 1, 0.1), histtype='stepfilled', align='mid', 
        color='black', linewidth=1, weights=weights, cumulative=False)
ax.set_xlim(-1, 1)

fig2, ax2 = plt.subplots()
cum_hist = ax2.hist(np.abs(shift_error), np.arange(0, 1.1, 0.001), histtype='step', align='mid', 
        color='black', linewidth=1, weights=weights, cumulative=True)
ax2.axhline(0.95, linestyle='--', color='red')
ax2.set_xlim(-0.01, 1)
'''