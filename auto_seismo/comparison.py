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

directory = '../../seismograms/seis_1/'
files = np.sort(os.listdir(directory))
seis_files = files[0:4]

keras.losses.huber_loss = huber_loss
arrive_model = load_model('./models/arrival_prediction_model.h5')

def make_pred(file, flip=False):
    seismogram = obspy.read(directory+file)[0]
    time = seismogram.times()
    init = np.where(time > seismogram.stats.sac.t2 + np.random.rand()*8 - 10 - seismogram.stats.sac.b)[0][0]
    end = init + 1000
    time = time[init:end] - time[init]
    amp_i = seismogram.data[init:end]
    if flip:
        amp_i = -amp_i
    amp_i = (amp_i - amp_i.min()) / (amp_i.max() - amp_i.min())
    pred = arrive_model.predict(amp_i[:400].reshape((1, 400, 1))).flatten()[0]
    actual = seismogram.stats.sac.t6 - seismogram.stats.sac.b - seismogram.times()[init]
    
    return time, amp_i, pred, actual

plot_flip = False
save_gif = False

fig, ax = plt.subplots()
seis_plot = ax.plot([], [])[0]
pred_line = ax.plot([], [], color='black', linestyle='--', label='Prediction')[0]
actual_line = ax.plot([], [], color='red', label='Actual')[0]
ax.set_xlim(0, 100)
ax.set_ylim(-0.05, 1.05)
ax.xaxis.set_minor_locator(mtick.MultipleLocator(5))
ax.yaxis.set_minor_locator(mtick.MultipleLocator(0.1))
ax.set_xlabel('Time From Cut [s]')
ax.set_ylabel('Relative Amplitude')
ax.legend()

def init():
    seis_plot.set_data([],[])
    pred_line.set_data([],[])
    actual_line.set_data([],[])
    return seis_plot

def animate_noflip(i):
    file = seis_files[3]
    time, amp_i, pred, actual = make_pred(file)
    
    seis_plot.set_data(time, amp_i)
    pred_line.set_data(np.ones(50)*pred, np.linspace(-0.05, 1.05))
    actual_line.set_data(np.ones(50)*actual, np.linspace(-0.05, 1.05))
    return seis_plot

def animate_flip(i):
    file = seis_files[3]
    time, amp_i, pred, actual = make_pred(file, flip=True)
    
    seis_plot.set_data(time, amp_i)
    pred_line.set_data(np.ones(50)*pred, np.linspace(-0.05, 1.05))
    actual_line.set_data(np.ones(50)*actual, np.linspace(-0.05, 1.05))
    return seis_plot

if not plot_flip:
    ax.set_title('Unflipped Seismogram')
    plt.tight_layout()
    animated_noflip = animation.FuncAnimation(fig, animate_noflip, init_func=init,
                                        frames=10, interval=1000)
    if save_gif:
        animated_noflip.save('../figs/unflipped_seis_pred.gif', 
                             writer=animation.PillowWriter(fps=1))
else:
    ax.set_title('Flipped Seismogram')
    plt.tight_layout()
    animated_flip = animation.FuncAnimation(fig, animate_flip, init_func=init,
                                        frames=10, interval=1000)
    if save_gif:
        animated_flip.save('../figs/flipped_seis_pred.gif',
                           writer=animation.PillowWriter(fps=1))

'''
# Compare different files preds
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
        plt.tight_layout()
'''

'''
# Compare same file diff window predictions
for i in range(10):
    file = seis_files[0]
    seismogram = obspy.read(directory+file)[0]
    time = seismogram.times()
    init = np.where(time > seismogram.stats.sac.t2 + np.random.rand()*8 - 10 - seismogram.stats.sac.b)[0][0]
    end = init + 1000
    time = time[init:end] - time[init]
    amp_i = seismogram.data[init:end]
    if i > 4:
        amp_i = -amp_i
    amp_i = (amp_i - amp_i.min()) / (amp_i.max() - amp_i.min())
    pred = arrive_model.predict(amp_i[:400].reshape((1, 400, 1))).flatten()[0]
    actual = seismogram.stats.sac.t6 - seismogram.stats.sac.b - seismogram.times()[init]
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
    plt.tight_layout()
    plt.close()
'''