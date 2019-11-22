# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 20:02:17 2019

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
mpl.rcParams['xtick.major.size'] = 12
mpl.rcParams['xtick.minor.size'] = 8
mpl.rcParams['xtick.labelsize'] = 14
mpl.rcParams['ytick.major.size'] = 12
mpl.rcParams['ytick.minor.size'] = 8
mpl.rcParams['ytick.labelsize'] = 14

vespa = 'original'
file_path = '../autoencoder/' + vespa + '_vespa.txt'
#levels = np.asarray([-6000, -4500, -3000, -1500, 0, 1500, 3000, 4500, 6000])
# Found levels from plotting original first (contour = ax.contourf(..))
# and getting the proper attribute (contour.levels), use this throughout
# or just make a grid...

#parser = argparse.ArgumentParser(description='Create vespagram contour plot from txt file data.')
#parser.add_argument('file_path', metavar='File Path', type=str, default=file_path)

#args = parser.parse_args()

#file_path = args.file_path

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

step = 40
limit = 400
levels = np.arange(-limit, limit+step, step)
cmap = plt.get_cmap('seismic')
fig, ax = plt.subplots()
ax.set_title(file_path.split('/')[-1].rstrip('.txt'))
contour = ax.contourf(time, slowness, amplitude, levels, cmap=cmap, extend='both')
contour.cmap.set_under('k')
ax.invert_yaxis()
ax.set_ylabel('Slowness (s/deg)')
ax.set_xlabel('Travel time (s)')
ax.xaxis.set_major_locator(mtick.MultipleLocator(50))
ax.xaxis.set_minor_locator(mtick.MultipleLocator(25))
ax.yaxis.set_major_locator(mtick.MultipleLocator(0.2))
ax.yaxis.set_minor_locator(mtick.MultipleLocator(0.1))
fig.tight_layout(pad=0.5)
fig.savefig('../figs/' + file_path.split('/')[-1].rstrip('.txt') + '.svg', dpi=500)