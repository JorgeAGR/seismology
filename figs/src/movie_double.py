#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 21:21:23 2020

@author: jorgeagr
"""
import os
import argparse
import obspy
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.animation as animation

# Settings for plots
golden_ratio = (np.sqrt(5) + 1) / 2
width = 25
height = 10

mpl.rcParams['figure.figsize'] = (width, height)
mpl.rcParams['font.size'] = 20
mpl.rcParams['figure.titlesize'] = 'large'
mpl.rcParams['legend.fontsize'] = 'medium'
mpl.rcParams['xtick.major.size'] = 12
mpl.rcParams['xtick.minor.size'] = 8
mpl.rcParams['xtick.labelsize'] = 16
mpl.rcParams['ytick.major.size'] = 12
mpl.rcParams['ytick.minor.size'] = 8
mpl.rcParams['ytick.labelsize'] = 16
plt.rcParams['animation.ffmpeg_path'] ='/usr/bin/ffmpeg'

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
                         '20130721A.newzealand.S49A.BHT.s_fil', '-s', '-ncomparison',
                         '-b55', '-e85'])

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
#end_time = -np.abs(end_t)
end_time = np.abs(end_t)
end_time = np.round(end_time + shift, decimals=1)

time_i_grid = np.arange(begin_time, end_time - time_window, 0.1)
time_f_grid = np.arange(begin_time + time_window, end_time, 0.1)

window_preds = scan(seis, times, time_i_grid, time_f_grid, shift, model)
neg_preds = scan(-seis, times, time_i_grid, time_f_grid, shift, model)
times = times - shift

fig, ax = plt.subplots(nrows=2, sharex=True)#figsize=(15,5))
plot_dic = {'win_init0': None, 'win_end0': None, 'pred0': None,
            'win_init1': None, 'win_end0': None, 'pred1': None}
data = {0: window_preds, 1: neg_preds}
for i in range(2):
    ax[i].set_ylim(-1, 1)
    #ax[i].set_xlim(times.min(), times.max())
    ax[i].set_xlim(-begin_t, end_t)
    ax[i].set_xlabel('Time (s)')
    ax[i].set_ylabel('Amplitude')
    ax[i].xaxis.set_major_locator(mtick.MultipleLocator(50))
    ax[i].xaxis.set_minor_locator(mtick.MultipleLocator(25))
    ax[i].yaxis.set_major_locator(mtick.MultipleLocator(0.5))
    ax[i].yaxis.set_minor_locator(mtick.MultipleLocator(0.25))
    ax[i].plot(times, seis*(-1)**i, color='black')
    plot_dic['win_init{}'.format(i)] = ax[i].axvline(time_i_grid[0]-shift, color='gray')
    plot_dic['win_end{}'.format(i)] = ax[i].axvline(time_f_grid[0]-shift, color='gray')
    plot_dic['pred{}'.format(i)] = ax[i].axvline(data[i][0]-shift, color='red', linestyle='--')

fig.tight_layout(pad=0.5)
def animate(i):
    i = i
    plot_dic['win_init0'].set_xdata(time_i_grid[i]-shift)
    plot_dic['win_end0'].set_xdata(time_f_grid[i]-shift)
    plot_dic['pred0'].set_xdata(window_preds[i]-shift)
    plot_dic['win_init1'].set_xdata(time_i_grid[i]-shift)
    plot_dic['win_end1'].set_xdata(time_f_grid[i]-shift)
    plot_dic['pred1'].set_xdata(neg_preds[i]-shift)

anim = animation.FuncAnimation(fig, animate, interval=1,
                               frames=window_preds.shape[0], repeat=True)
if save:
    anim.save('../{}.mp4'.format(name), writer=animation.FFMpegWriter(fps=60))
