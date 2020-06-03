#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 23:04:31 2020

@author: jorgeagr
"""

import os
import argparse
import obspy
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.cluster import DBSCAN
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.animation as animation

# Settings for plots
golden_ratio = (np.sqrt(5) + 1) / 2
width = 25
height = 15

mpl.rcParams['figure.figsize'] = (width, height)
mpl.rcParams['font.size'] = 22
mpl.rcParams['figure.titlesize'] = 'large'
mpl.rcParams['legend.fontsize'] = 'medium'
mpl.rcParams['xtick.major.size'] = 16
mpl.rcParams['xtick.major.width'] = 2
mpl.rcParams['xtick.minor.width'] = 2
mpl.rcParams['xtick.minor.size'] = 12
mpl.rcParams['xtick.labelsize'] = 36
mpl.rcParams['ytick.major.size'] = 16
mpl.rcParams['ytick.major.width'] = 2
mpl.rcParams['ytick.minor.size'] = 12
mpl.rcParams['ytick.minor.width'] = 2
mpl.rcParams['ytick.labelsize'] = 36
mpl.rcParams['axes.linewidth'] = 2
plt.rcParams['animation.ffmpeg_path'] ='/usr/bin/ffmpeg'

# Most of script copied from movie template. Will change into functional parser later.
parser = argparse.ArgumentParser(description='Plot a seismogram in a file directory. Plots the first seismogram in the directory by default.')
parser.add_argument('file_dir', help='SAC files directory.', type=str)
parser.add_argument('file', help='Name of specific file to plot.', type=str)
parser.add_argument('-i', '--index', help='Index of file to plot.', type=int, default=0)
parser.add_argument('-s', '--save', help='Save the movie as mp4.', action='store_true')
parser.add_argument('-n', '--savename', help='Name to save the movie under.', type=str)
parser.add_argument('-b', '--begin', help='Seconds before theoretical arrival to begin prediction.', type=float,
                    default=-300)
parser.add_argument('-e', '--end', help='Seconds before theoretical arrival to end prediction.', type=float,
                    default=-100)
args = parser.parse_args(['/home/jorgeagr/Documents/seismograms/SS_kept/',
                         '20160101A.india.DRLN.BHT.s_fil', '-s', '-nscanning_eg',
                         '-b40', '-e90'])

file_dir = args.file_dir
file = args.file
index = args.index
save = args.save
name = args.savename
begin_t = args.begin
end_t = args.end
resample_Hz = 10

if not file:
    file = sorted(os.listdir(file_dir))[index]
extension = '.{}'.format(file.split('.')[-1])

if not name:
    name = file.rstrip('.'+extension)

def cut_Window(cross_sec, times, t_i, t_f):
    #init = np.where(times == np.round(t_i, 1))[0][0]
    #end = np.where(times == np.round(t_f, 1))[0][0]
    init = int(np.round(t_i*resample_Hz))
    end = int(np.round(t_f*resample_Hz))
    
    return cross_sec[init:end]

def scan(seis, times, time_i_grid, time_f_grid, shift, model, negative=False):
    window_preds = np.zeros(len(time_i_grid))
    for i, t_i, t_f in zip(range(len(time_i_grid)), time_i_grid, time_f_grid):
        seis_window = cut_Window(seis, times, t_i, t_f) * (-1)**negative
        seis_window = seis_window / np.abs(seis_window).max()
        # Take the absolute value of the prediction to remove any wonky behavior in finding the max
        # Doesn't matter since they are bad predictions anyways
        window_preds[i] += np.abs(model.predict(seis_window.reshape(1, len(seis_window), 1))[0][0]) + t_i
    return window_preds

model = load_model('../../pickerlite/models/SS_40_model.h5')
seismogram = obspy.read(file_dir+file)[0]
times = seismogram.times()
time_window = 40

seis = seismogram.data / np.abs(seismogram.data).max()
shift = -seismogram.stats.sac.b
begin_time = -np.abs(begin_t)
begin_time = np.round(begin_time + shift, decimals=1)
end_time = -np.abs(end_t)
end_time = np.abs(end_t)
end_time = np.round(end_time + shift, decimals=1)

time_i_grid = np.arange(begin_time, end_time - time_window, 0.1)
time_f_grid = np.arange(begin_time + time_window, end_time, 0.1)

window_preds = scan(seis, times, time_i_grid, time_f_grid, shift, model)
times = times - shift

grouped_preds = window_preds
dbscan = DBSCAN(0.05, 2)
dbscan.fit(window_preds.reshape(-1,1))
clusters, counts = np.unique(dbscan.labels_, return_counts=True)
for c in clusters:
    if c != -1:
        grouped_preds[np.where(dbscan.labels_ == c)] = np.ones_like(grouped_preds[dbscan.labels_ == c]) * grouped_preds[dbscan.labels_ == c].mean()

steps = [0, 150, 300]
fig, axis = plt.subplots(nrows=5, sharex=True, gridspec_kw={'hspace': 0,
                                                            'height_ratios': [1, 1, 1, 0.4, 2]})#figsize=(15,5))
for n, ax in enumerate(axis[:-2]):
    ax.set_ylim(-1, 1)
    ax.set_xlim(-begin_t, end_t)
    ax.yaxis.set_major_locator(mtick.MultipleLocator(1))
    ax.yaxis.set_minor_locator(mtick.MultipleLocator(0.5))
    ax.plot(times, seis, color='black')
    win_init = ax.axvline(time_i_grid[steps[n]]-shift, color='gray', label='Window Limit')
    win_end = ax.axvline(time_f_grid[steps[n]]-shift, color='gray')
    y_grid = np.array([-1, 1])
    win_col = ax.fill_betweenx(y_grid, np.ones_like(y_grid)*(time_i_grid[steps[n]]-shift),
                               np.ones_like(y_grid)*(time_f_grid[steps[n]]-shift), color='whitesmoke')
    pred = ax.axvline(window_preds[steps[n]]-shift, color='red', linestyle='--', label='Arrival Prediction')
    #ax.legend()

axis[3].set_visible(False)

axis[1].set_ylabel('Amplitude', fontsize=36)
axis[-1].set_xlabel('Time (s)', fontsize=36)
axis[-1].xaxis.set_major_locator(mtick.MultipleLocator(20))
axis[-1].xaxis.set_minor_locator(mtick.MultipleLocator(10))
axis[0].legend(loc='upper right')

axis[-1].hist(grouped_preds-shift, np.arange(begin_time, end_time + 0.1, 0.1)-shift,
    color='red', weights=np.ones(len(window_preds)) / 400, linewidth=2)
axis[-1].set_ylabel('Quality', fontsize=36)
axis[-1].yaxis.set_major_locator(mtick.MultipleLocator(0.2))
axis[-1].yaxis.set_minor_locator(mtick.MultipleLocator(0.1))

fig.tight_layout(pad=0.5)
fig.savefig('../{}.pdf'.format(name), dpi=200)
