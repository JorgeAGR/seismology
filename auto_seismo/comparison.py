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

directory = '../seismograms/seis_1/'
files = np.sort(os.listdir(directory))
seis_files = files[0:4]

keras.losses.huber_loss = huber_loss
arrive_model = load_model('./models/arrival_prediction_model_noflip.h5')

for flip in range(2):
    for file in seis_files:
        seismogram = obspy.read(directory+file)[0]
        time = seismogram.times()
        init = np.where(time > seismogram.stats.sac.t2 - 10 - seismogram.stats.sac.b)[0][0]
        end = init + 1000
        time = time[init:end] - time[init]
        amp_i = seismogram.data[init:end]
        if flip:
            amp_i = -amp_i
        amp_i = (amp_i - amp_i.min()) / (amp_i.max() - amp_i.min())
        pred = arrive_model.predict(amp_i[:400].reshape((1, 400, 1))).flatten()[0]
        actual = seismogram.stats.sac.t6 - seismogram.stats.sac.t2 + 10
        fig, ax = plt.subplots()
        ax.plot(time, amp_i)
        ax.axvline(pred, color='black', linestyle='--', label='Prediction')
        ax.axvline(actual, color='red', label='Actual')
        ax.set_xlim(0, 100)
        ax.set_ylim(-0.05, 1.05)
        ax.xaxis.set_minor_locator(mtick.MultipleLocator(5))
        ax.yaxis.set_minor_locator(mtick.MultipleLocator(0.1))
        ax.set_xlabel('Time From Cut [s]')
        ax.set_ylabel('Relative Amplitude')


'''
fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True)
#sample_seis = np.random.randint(0, high=len(seismograms), size=4)
#sample_seis = np.array([34982, 197662, 22390, 165919])
# good one: 34982, 197662, 22390, 165919
sample_seis = np.array([0, 1, 2, 3])
#t = np.arange(0, 40, 0.1)
for n, i in enumerate(sample_seis):
    s = obspy.read(datadir+files[i])[0]
    t = s.times()
    init = np.where(t > s.stats.sac.t2 - 10 - s.stats.sac.b)[0][0]
    end = init + 1000
    t = t[init:end] - t[init]
    amp_i = s.data[init:end]
    amp_i = (amp_i - amp_i.min()) / (amp_i.max() - amp_i.min())
    p = arrive_model.predict(amp_i[:400].reshape((1, 400, 1))).flatten()[0]
    a = s.stats.sac.t6 - s.stats.sac.t2 + 10
    ax[n//2][n%2].plot(t, amp_i, color='black')
    ax[n//2][n%2].axvline(p, color='black', linestyle='--', label='Prediction')
    ax[n//2][n%2].axvline(a, color='red', label='Actual')
    ax[n//2][n%2].set_xlim(0, 100)
    ax[n//2][n%2].set_ylim(-0.05, 1.05)
    ax[n//2][n%2].xaxis.set_minor_locator(mtick.MultipleLocator(5))
    ax[n//2][n%2].yaxis.set_minor_locator(mtick.MultipleLocator(0.1))
    #ax[1][n%2].set_xlabel('Time From Cut [s]')
    #ax[n//2][n%1].set_ylabel('Relative Amplitude')
ax[0][1].legend(fontsize=12)
fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.xlabel('Time From Cut [s]')
plt.ylabel('Relative Amplitude')
#fig.text(0.5, 0.02, 'Time From Cut [s]', ha='center')
#fig.text(0.04, 0.5, 'Relative Amplitude', va='center', rotation='vertical')
plt.tight_layout()
'''