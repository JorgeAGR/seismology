#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 20:29:44 2019

@author: jorgeagr
"""
import os
import obspy
import keras
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.animation as animation
from keras.models import load_model
from tensorflow.losses import huber_loss

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

directory = '../../seismograms/SS_kept_test/'
dir_name = directory.split('/')[-2] + '/'
try:
    os.makedirs('../figs/etc/' + dir_name)
except:
    pass

files = np.sort(os.listdir(directory))
seis_files = files

for file in seis_files:
    seismogram = obspy.read(directory+file)[0]
    time = seismogram.times()
    init = np.where(time > (seismogram.stats.sac.t2 - 100 - seismogram.stats.sac.b))[0][0]
    end = len(time)#np.where(time > (seismogram.stats.sac.t2 + 100 - seismogram.stats.sac.b))[0][0]
    shift = time[init]
    time = time[init:end]# - time[init]
    amp_i = seismogram.data[init:end]
    #amp_i = (amp_i - amp_i.min()) / (amp_i.max() - amp_i.min())
    amp_i = amp_i / np.abs(amp_i).max()
    # for test data
    pred = seismogram.stats.sac.t7
    actual = seismogram.stats.sac.t6 - seismogram.stats.sac.b
    # if using unknown data
    #pred = seismogram.stats.sac.t6 #- shift
    theoretical = seismogram.stats.sac.t2 - seismogram.stats.sac.b
    fig, ax = plt.subplots()
    ax.plot(time, amp_i)
    ax.axvline(theoretical, color='black', linestyle='--', label='th')
    if actual:
        ax.axvline(actual, color='red', linestyle='--', label='actual')
    ax.axvline(pred, color='red', label='pred')
    ax.set_xlim(time[0], time[-1])
    ax.set_ylim(-1.05, 1.05)
    ax.xaxis.set_minor_locator(mtick.MultipleLocator(5))
    ax.yaxis.set_minor_locator(mtick.MultipleLocator(0.1))
    ax.set_xlabel('Time From Cut [s]')
    ax.set_ylabel('Relative Amplitude')
    plt.tight_layout()
    plt.close()
    fig.savefig('../figs/etc/' + dir_name + 'pred_' + file + '.svg', dpi=500)