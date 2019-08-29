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

shifting_npz = np.load('../test/results_train_data.npz')
files = shifting_npz['files'].astype(np.str)
pred_avg = shifting_npz['pred_avg']
pred_err = shifting_npz['pred_err']
flipped = shifting_npz['flipped']

if 'train_data_arrivals.npz' not in os.listdir('./train_data/etc/'):
    dir_files, arrivals = test_pred('../../seismograms/SS_kept/')
    np.savez('./train_data/etc/train_data_arrivals.npz', files=files, arrivals=arrivals)
else:
    actuals_npz = np.load('./train_data/etc/train_data_arrivals.npz')
    dir_files = actuals_npz['files'].astype(np.str)
    arrivals = actuals_npz['arrivals']

avail_files = []
avail_arrivals = []
for i, file in enumerate(dir_files):
    if str(file[:-6]) in files:
        avail_files.append(file)
        avail_arrivals.append(arrivals[i])
dir_files = np.asarray(avail_files)
arrivals = np.asarray(avail_arrivals)