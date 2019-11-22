# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 21:52:01 2019

@author: jorge
"""

import os
import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# Settings for plots
golden_ratio = (np.sqrt(5) + 1) / 2
width = 15
height = width / golden_ratio

mpl.rcParams['figure.figsize'] = (width, height)
mpl.rcParams['font.size'] = 14
mpl.rcParams['figure.titlesize'] = 'large'
mpl.rcParams['legend.fontsize'] = 'small'
mpl.rcParams['xtick.major.size'] = 8
mpl.rcParams['xtick.minor.size'] = 4
mpl.rcParams['xtick.labelsize'] = 14
mpl.rcParams['ytick.major.size'] = 8
mpl.rcParams['ytick.minor.size'] = 4
mpl.rcParams['ytick.labelsize'] = 14


def create_Meshgrid(file_path):    
    vespa_data = np.loadtxt(file_path)
    time = vespa_data[:,0]
    slowness = vespa_data[:,1]
    amplitude = vespa_data[:,2]
    
    _, unique_vals_counts = np.unique(slowness, return_counts=True)
    x_dim = len(unique_vals_counts)
    y_dim = unique_vals_counts[0]
    
    time = time.reshape(x_dim, y_dim)
    slowness = slowness.reshape(x_dim, y_dim)
    amplitude = amplitude.reshape(x_dim, y_dim)
    
    return time, slowness, amplitude

vespas = ['3sigma', '2sigma', '4x', '16x']
letters = ['a)', 'b)', 'c)', 'd)']
step = 40
limit = 400
levels = np.arange(-limit, limit+step, step)
cmap = plt.get_cmap('seismic')
fig, ax = plt.subplots(nrows=2, ncols=2)
for i in range(4):
    file_path = vespas[i] + '_vespa.txt'
    time, slowness, amplitude = create_Meshgrid(file_path)
    contour = ax[i//2][i%2].contourf(time, slowness, amplitude, levels, cmap=cmap, extend='both')
    contour.cmap.set_under('k')
    ax[i//2][i%2].invert_yaxis()
    ax[i//2][i%2].set_ylabel('Slowness (s/deg)')
    ax[i//2][i%2].set_xlabel('Travel time (s)')
    ax[i//2][i%2].xaxis.set_major_locator(mtick.MultipleLocator(50))
    ax[i//2][i%2].xaxis.set_minor_locator(mtick.MultipleLocator(25))
    ax[i//2][i%2].yaxis.set_major_locator(mtick.MultipleLocator(0.2))
    ax[i//2][i%2].yaxis.set_minor_locator(mtick.MultipleLocator(0.1))
    ax[i//2][i%2].text(-380, -1.8, letters[i], fontsize='x-large', fontweight='bold')
fig.tight_layout(pad=0.5)
fig.savefig('../figs/vespagrams_results.eps', dpi=500)