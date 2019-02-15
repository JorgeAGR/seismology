#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 16:03:26 2018

@author: jorgeagr
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from scipy.stats import norm

def gaussian(x, mu, std):
        return np.exp(-(x - mu)**2 / (2 * std ** 2))

golden_ratio = (np.sqrt(5) + 1) / 2
width = 12
height = width / golden_ratio

mpl.rcParams['font.size'] = 12
mpl.rcParams['figure.titlesize'] = 'large'
mpl.rcParams['legend.fontsize'] = 'small'
mpl.rcParams['figure.figsize'] = (width, height)

seismograms = np.load('seismograms.npy')
arrivals = np.load('arrivals.npy')
pred = np.load('predictions.npy')

error = np.abs(arrivals - pred)

density = False
mean, std = norm.fit(error)

fig, ax = plt.subplots()
ticks = np.arange(0, 7, 1)
ticks = np.append(ticks, 1.8)

counts, bins, _ = ax.hist(error, np.arange(0, 6, 0.2), align='mid', density=density,
                          histtype='step', color='black', linewidth=2)
#x = np.linspace(0.5, bins[-1], num=100)
#ax.plot(x-1, norm.pdf(x, mean, std),)
ax.set_xlim(0, 3)
ax.xaxis.set_major_locator(mtick.MultipleLocator(1))
ax.xaxis.set_minor_locator(mtick.MultipleLocator(0.2))
ax.set_xlabel('Arrival Time Error [s]')
ax.set_ylabel('Number of picks')
if density == True:
    ax.yaxis.set_major_locator(mtick.MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(mtick.MultipleLocator(0.02))
else:
    ax.yaxis.set_major_locator(mtick.MultipleLocator(25000))
    ax.yaxis.set_minor_locator(mtick.MultipleLocator(5000))
plt.tight_layout()
#fig.savefig('picking_his')