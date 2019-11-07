#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 15:56:19 2019

@author: jorgeagr
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import pandas as pd
from mpl_toolkits.basemap import Basemap

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


def get_Lauren_Pred(cap, precursor):
    file_path = 'cross_secs_dat/lauren_pred/' + cap + 'caps_S' + precursor + 'S.dat'
    with open(file_path) as datfile:
        dat_bins = np.asarray([line.split(' ')[0] for line in datfile])
    dat_times = np.loadtxt(file_path, usecols=(1, 2))
    return dat_bins, dat_times[:,0], dat_times[:,1]

def get_Model_Pred(cap, precursor):
    file_path = 'cross_secs_dat/model_pred/' + cap + 'caps_wig_preds.csv'
    df = pd.read_csv(file_path)
    df = df.loc[:, ['file', precursor+'pred', precursor+'err', precursor+'amp', precursor+'qual']]
    
    '''
    Line for now. Will likeyl remove the .sac from pred file in the future.
    UPDATE: Actual script removes extension twice. It caused weid ordering in the
    files. Remove once new file is generated. See below as well.
    '''
    df.loc[:, 'file'] = [string.rstrip('.sac') for string in df['file'].values]
    
    '''
    Temporary sorting indeces. Remove from below and return once new file is
    generated.
    '''
    ind = np.argsort(df['file'].values)
    
    return df['file'].values[ind], df[precursor+'pred'].values[ind], df[precursor+'err'].values[ind]

def cap2latlon(bins):
    # Converts string into floats (in radians)
    # Then from radians to degrees
    latlon = np.zeros((len(bins), 2))
    for i, bin in enumerate(bins):
        neg = 0
        lat, lon = bin.split('_')
        if lat[0] == 'n':
            neg = 1
        lat = (-1)**neg * float(lat[neg:])
        lon = float(lon)
        latlon[i,0] = np.rad2deg(lat)
        latlon[i,1] = np.rad2deg(lon)
    return latlon

def get_MinMax_Times(l_times, m_times):
    min_time = l_times.min()
    max_time = l_times.max()
    
    model_min = m_times.min()
    model_max = m_times.max()
    
    if model_min < min_time:
        min_time = model_min
    if model_max > max_time:
        max_time = model_max
    
    return min_time, max_time

discontinuity = '660'
cap = '15'

lauren_bins, lauren_times, lauren_errors = get_Lauren_Pred(cap, discontinuity)
lauren_latlon = cap2latlon(lauren_bins)

'''
TEMPORARY. REMOVE WHEN LAUREN FIXES
'''
lauren_times = lauren_times-4.2

model_bins, model_times, model_errors = get_Model_Pred(cap, discontinuity)
model_latlon = cap2latlon(model_bins)

#fig, ax = plt.subplots(nrows=3, ncols=2)
fig = plt.figure()
rows = 13
cbar_span = 1
cols = rows - cbar_span
span = cols//2
ax = [[plt.subplot2grid((rows,cols), (0,0), colspan=span, rowspan=span, fig=fig),
      plt.subplot2grid((rows,cols), (0,span), colspan=span, rowspan=span, fig=fig)],
      plt.subplot2grid((rows,cols), (span,0), colspan=cols, rowspan=1, fig=fig),
      [plt.subplot2grid((rows,cols), (span+cbar_span,0), colspan=span, rowspan=span, fig=fig),
      plt.subplot2grid((rows,cols), (span+cbar_span,span), colspan=span, rowspan=span, fig=fig)]]
fig.text(0.46, 0.925, 'Model Map', fontweight='bold')
fig.text(0.45, 0.375, 'Bootstrap Map', fontweight='bold')
fig.text(0.475, 0.525, 'S'+discontinuity+'S', fontweight='bold')

cmap = mpl.cm.YlGnBu
min_time, max_time = get_MinMax_Times(lauren_times, model_times)
norm = mpl.colors.Normalize(vmin=min_time, vmax=max_time)
cbar = mpl.colorbar.ColorbarBase(ax[1], cmap=cmap, norm=norm, orientation='horizontal')
cbar.ax.tick_params(length=5)
cbar.set_label('Time Before Main Arrival [s]')
for l, lon_0 in enumerate([-180, 0]):
    globe_m = Basemap(projection='moll', lon_0=lon_0, resolution='c', ax=ax[0][l])
    globe_m.drawmapboundary(fill_color='lightgray')
    globe_m.drawcoastlines(color='gray')
    globe_m.drawparallels([-45, 0, 45])
    globe_m.drawmeridians(np.arange(0.,360.,45.))
    #globe.fillcontinents(color='black')
    #globe.scatter(lauren_latlon[85:,0], lauren_latlon[85:,1], color='red', s=10, latlon=True)
    globe_m.scatter(model_latlon[:,1], model_latlon[:,0], c=model_times, s=50, latlon=True, cmap=cmap, norm=norm, zorder=10)
    
    globe_l = Basemap(projection='moll', lon_0=lon_0, resolution='c', ax=ax[2][l])
    globe_l.drawmapboundary(fill_color='lightgray')
    globe_l.drawcoastlines(color='gray')
    globe_l.drawparallels([-45, 0, 45])
    globe_l.drawmeridians(np.arange(0.,360.,45.))
    #globe.fillcontinents(color='black')
    #globe.scatter(lauren_latlon[85:,0], lauren_latlon[85:,1], color='red', s=10, latlon=True)
    globe_l.scatter(lauren_latlon[:,1], lauren_latlon[:,0], c=lauren_times, s=50, latlon=True, cmap=cmap, norm=norm, zorder=10)
    #plt.show()
fig.tight_layout(pad=0.5)
'''
# figs showing time diffs
fig2, ax2 = plt.subplots()
for l, lon_0 in enumerate([0, 180]):
    diff_times = model_times - lauren_times
    globe_diff = Basemap(projection='moll', lon_0=lon_0, resolution='c', ax=ax2)
    globe_diff.drawmapboundary(fill_color='lightgray')
    globe_diff.drawcoastlines(color='gray')
    globe_diff.drawparallels([-45, 0, 45])
    globe_diff.drawmeridians(np.arange(0.,360.,45.))
    #globe.fillcontinents(color='black')
    #globe.scatter(lauren_latlon[85:,0], lauren_latlon[85:,1], color='red', s=10, latlon=True)
    globe_diff.scatter(model_latlon[:,1], model_latlon[:,0], c=diff_times, s=50, latlon=True, cmap='YlGnBu', zorder=10)
'''