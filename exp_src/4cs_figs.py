#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 17:18:23 2019

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

golden_ratio = (np.sqrt(5) + 1) / 2
width = 15
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

seis_files = np.sort(os.listdir(directory))
file = seis_files[0]

seismogram = obspy.read(directory+file)[0]
time = seismogram.times()

for i in range(2):
    fig1, ax1 = plt.subplots()
    ax1.plot(time, seismogram.data / np.abs(seismogram.data).max(), color='black')
    if i == 1:
        ax1.axvline(seismogram.stats.sac.t6 - seismogram.stats.sac.b, color='red', linestyle='--')
    ax1.set_xlim(time[0], time[-1])
    ax1.set_ylim(-1, 1)
    ax1.xaxis.set_minor_locator(mtick.MultipleLocator(10))
    ax1.yaxis.set_major_locator(mtick.MultipleLocator(0.5))
    ax1.yaxis.set_minor_locator(mtick.MultipleLocator(0.1))
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Amplitude')
    fig1.tight_layout()
    plt.savefig('../figs/4cs/seismogram_eg_' + str(i) + '.svg', dpi=500)

np.random.seed(seed=0)
np.random.shuffle(seis_files)
n_seis = 15
fig2, ax2 = plt.subplots(nrows=n_seis)
for i, ax in enumerate(ax2):
    seis = obspy.read(directory+seis_files[i])[0]
    ax.plot(time, seis.data / np.abs(seis.data).max(), color='black')
    ax.axvline(seis.stats.sac.t6 - seis.stats.sac.b, color='red', linestyle='--')
    ax.set_xlim(time[0], time[-1])
    ax.set_ylim(-1.1, 1.1)
    if i < n_seis - 1:
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())
    if i > 0:
        ax.spines['top'].set_visible(False)
ax.xaxis.set_minor_locator(mtick.MultipleLocator(10))
ax.yaxis.set_major_locator(plt.NullLocator())
ax.set_xlabel('Time [s]')
fig2.tight_layout(h_pad=0)
fig2.subplots_adjust(wspace=0, hspace=0)
plt.savefig('../figs/4cs/seismograms.svg', dpi=500)