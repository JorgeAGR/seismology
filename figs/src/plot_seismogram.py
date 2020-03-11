#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 16:38:14 2020

@author: jorgeagr
"""
import os
import argparse
import numpy as np
import obspy
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Settings for plots
golden_ratio = (np.sqrt(5) + 1) / 2
width = 25
height = 5

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

parser = argparse.ArgumentParser(description='Plot a seismogram in a file directory. Plots the first seismogram in the directory by default.')
parser.add_argument('file_dir', help='SAC files directory.', type=str)
parser.add_argument('-n', '--name', help='Name of specific file to plot.', type=str)
parser.add_argument('-i', '--index', help='Index of file to plot.', type=int, default=0)
parser.add_argument('-a', '--arrival', help='Plot the arrival.', action='store_true')
parser.add_argument('-s', '--save', help='Save the plot.', action='store_true')
parser.add_argument('-sn', '--savename', help='Name to save the plot under.', type=str)
args = parser.parse_args()

file_dir = args.file_dir
file = args.name
index = args.index
arrival = args.arrival
save = args.save
name = args.savename
if not args.savename:
    name = file

if not file:
    file = sorted(os.listdir(file_dir))[index]
extension = '.{}'.format(file.split('.')[-1])

seismogram = obspy.read(file_dir+file)[0]
time = seismogram.times()

fig, ax = plt.subplots()
ax.set_title(file.rstrip('.'+extension))
ax.plot(time, seismogram.data / np.abs(seismogram.data).max(), color='black')
if arrival:
    ax.axvline(seismogram.stats.sac.t6 - seismogram.stats.sac.b, color='red', linestyle='--')
ax.set_xlim(time[0], time[-1])
ax.set_ylim(-1, 1)
ax.xaxis.set_minor_locator(mtick.MultipleLocator(10))
ax.yaxis.set_major_locator(mtick.MultipleLocator(0.5))
ax.yaxis.set_minor_locator(mtick.MultipleLocator(0.25))
ax.set_xlabel('Time (s)')
ax.set_ylabel('Amplitude')
fig.tight_layout()
if save:
    fig.savefig('../{}.png'.format(file.rstrip('.'+extension)), dpi=250)
else:
    plt.show()