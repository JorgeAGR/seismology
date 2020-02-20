# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 22:15:54 2019

@author: jorge
"""

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# Settings for plots
golden_ratio = (np.sqrt(5) + 1) / 2
width = 12
height = width / golden_ratio

mpl.rcParams['figure.figsize'] = (width, height)
mpl.rcParams['font.size'] = 14
mpl.rcParams['figure.titlesize'] = 'large'
mpl.rcParams['legend.fontsize'] = 'small'
mpl.rcParams['xtick.major.size'] = 8
mpl.rcParams['xtick.minor.size'] = 4
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.major.size'] = 8
mpl.rcParams['ytick.minor.size'] = 4
mpl.rcParams['ytick.labelsize'] = 12

og_data = np.load('data/test/test_seismos.npy')
times = np.arange(0, 500, 0.1) - 400
fig, ax = plt.subplots(figsize=(width,3))
ax.plot(times, og_data[0], 'k')
ax.xaxis.set_major_locator(mtick.MultipleLocator(100))
ax.xaxis.set_minor_locator(mtick.MultipleLocator(50))
ax.yaxis.set_major_locator(plt.NullLocator())
#ax.yaxis.set_major_locator(mtick.MultipleLocator(0.5))
#ax.yaxis.set_minor_locator(mtick.MultipleLocator(0.25))
ax.set_xlim(times.min(), times.max())
ax.set_ylim(-1, 1)
ax.set_xlabel('Travel time (s)')
fig.tight_layout(pad=0.5)
fig.savefig('../figs/cae_project/seismo_original.eps', dpi=500)