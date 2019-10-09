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

def relu(x):
    return x * (x > 0)

x_grid = np.linspace(-2, 2, num=500)
figrelu, axrelu = plt.subplots(figsize=(5,5))
axrelu.plot(x_grid, relu(x_grid), color='black')
axrelu.xaxis.set_major_locator(mtick.MultipleLocator(1))
axrelu.xaxis.set_minor_locator(mtick.MultipleLocator(0.2))
axrelu.yaxis.set_major_locator(mtick.MultipleLocator(1))
axrelu.yaxis.set_minor_locator(mtick.MultipleLocator(0.2))
axrelu.spines['top'].set_visible(False)
axrelu.spines['right'].set_visible(False)
axrelu.set_xlim(-2, 2)
axrelu.set_ylim(-0.1, 2)
axrelu.set_xlabel('x')
axrelu.set_ylabel('y')
figrelu.tight_layout()
plt.savefig('../figs/4cs/relu.png', dpi=250)
plt.close()

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
    plt.savefig('../figs/4cs/seismogram_eg_' + str(i) + '.png', dpi=250)
    plt.close()

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
plt.savefig('../figs/4cs/seismograms.png', dpi=250)
plt.close()

fig3, ax3 = plt.subplots(figsize=(15,5))
ax3.plot(time, seismogram.data / np.abs(seismogram.data).max(), color='black')
ax3.axvline(seismogram.stats.sac.t2 - seismogram.stats.sac.b,
            color='gray', linestyle='--', label='Theory')
ax3.axvline(seismogram.stats.sac.t2 - seismogram.stats.sac.b + 30,
            color='gray', label='Window')
ax3.axvline(seismogram.stats.sac.t6 - seismogram.stats.sac.b,
            color='red', label='Picked')
ax3.set_xlim(300, time[-1])
ax3.set_ylim(-1, 1)
ax3.xaxis.set_major_locator(mtick.MultipleLocator(100))
ax3.xaxis.set_minor_locator(mtick.MultipleLocator(10))
ax3.yaxis.set_major_locator(plt.NullLocator())
ax3.set_xlabel('Time [s]')
ax3.legend(loc='upper left')
fig3.tight_layout(h_pad=0, w_pad=0)
plt.savefig('../figs/4cs/seismogram_window.png', dpi=250)
plt.close()

rand_shifts = [0, -3.3, -2.5, -1.2, 3, 4.6]
rand_shifts[0] = 0
figrand = plt.figure(figsize=(15,5))
axrand = [plt.subplot2grid((2,3), (0,0), colspan=1, rowspan = 1, fig=figrand),
          plt.subplot2grid((2,3), (0,1), colspan=1, rowspan = 1, fig=figrand),
          plt.subplot2grid((2,3), (0,2), colspan=1, rowspan = 1, fig=figrand),
          plt.subplot2grid((2,3), (1,0), colspan=1, rowspan = 1, fig=figrand),
          plt.subplot2grid((2,3), (1,1), colspan=1, rowspan = 1, fig=figrand),
          plt.subplot2grid((2,3), (1,2), colspan=1, rowspan = 1, fig=figrand),]

for i, ax in enumerate(axrand):
    rand_arrival = seismogram.stats.sac.t2 - seismogram.stats.sac.b - rand_shifts[i]
    init = np.where(np.round(rand_arrival - 40, 1) == time)[0][0]
    end = np.where(np.round(rand_arrival + 40, 1) == time)[0][0]
    amp = seismogram.data[init:end]
    amp = amp / np.abs(amp).max()
    
    ax.plot(time[init:end], amp, color='black')
    ax.axvline(seismogram.stats.sac.t6 - seismogram.stats.sac.b - time[init],
                color='red', label='Picked')
    ax.set_xlim(time[init], time[end])
    ax.set_ylim(-1, 1)
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())
figrand.tight_layout()
figrand.subplots_adjust(wspace=0, hspace=0)
plt.savefig('../figs/4cs/windows_shifted.png', dpi=250)
plt.close()

model_windows = [20, 25, 30, 40, 50]
loss_dir = '../auto_seismo/models/etc/'
loss_files = np.sort([model for model in os.listdir(loss_dir) if 'pos' in model])
loss_matrix = np.zeros((len(loss_files), 40))
val_loss_matrix = np.zeros((len(loss_files), 40))

min_loss = np.zeros((2, len(loss_files)))
min_val_loss = np.zeros((2, len(loss_files)))

for i, file in enumerate(loss_files):
    model_loss_npz = np.load(loss_dir + file)
    tr_l = model_loss_npz['loss'].mean(axis=0) # Average per epoch
    tr_l_min = model_loss_npz['loss'].min(axis=1) # Average of min loss
    loss_matrix[i,:len(tr_l)] = tr_l
    min_loss[0,i] = tr_l_min.mean()
    min_loss[1,i] = tr_l_min.std()
    v_l = model_loss_npz['val_loss'].mean(axis=0)
    v_l_min = model_loss_npz['val_loss'].min(axis=1)
    val_loss_matrix[i,:len(v_l)] = v_l
    min_val_loss[0,i] = v_l_min.mean()
    min_val_loss[1,i] = v_l_min.std()

figloss, axloss = plt.subplots()
axloss.errorbar(model_windows, min_loss[0,:], yerr=min_loss[1,:],
                color='blue', capsize=5, label='Training')
axloss.errorbar(model_windows, min_val_loss[0,:], yerr=min_val_loss[1,:],
                linestyle='--', color='red', capsize=5, label='Testing')
#axloss.set_xlim(19, 51)
axloss.set_ylim(0.01, 0.09)
axloss.yaxis.set_major_locator(mtick.MultipleLocator(0.02))
axloss.yaxis.set_minor_locator(mtick.MultipleLocator(0.01))
axloss.set_xlabel('Window Size [s]')
axloss.set_ylabel('Minimum Loss')
axloss.legend()
figloss.tight_layout(h_pad=0, w_pad=0)
plt.savefig('../figs/4cs/min_loss.png', dpi=250)
plt.close()

windows = [20, 40]
figwin = plt.figure(figsize=(10, 8))
axwin = [plt.subplot2grid((2,1), (0,0), colspan=1, rowspan = 1, fig=figwin),
          plt.subplot2grid((2,1), (1,0), colspan=1, rowspan = 1, fig=figwin)]
for i, ax in enumerate(axwin):
    ax.xaxis.set_major_locator(plt.NullLocator())
    init = np.where(np.round(seismogram.stats.sac.t2 - seismogram.stats.sac.b, 1) < time)[0][0]
    end = np.where(np.round(seismogram.stats.sac.t2 - seismogram.stats.sac.b + windows[i], 1) == time)[0][0]
    amp = seismogram.data[init:end]
    amp = amp / np.abs(amp).max()
    
    ax.set_title(str(windows[i]) + 'seconds')
    ax.plot(time[init:end], amp, color='black')
    ax.set_ylim(-1.1, 1.1)
figwin.tight_layout()
figwin.subplots_adjust(wspace=0, hspace=0.8)
plt.savefig('../figs/4cs/window_grid.png', dpi=250)
plt.close()