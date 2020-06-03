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
import matplotlib.ticker as mtick
import pandas as pd
from mpl_toolkits.basemap import Basemap
import argparse

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

# Safegaurd when running on IDE
discontinuity = '660'#'410'
cap = '7.5'

parser = argparse.ArgumentParser(description='Model vs Lauren Discontinuity Map')
parser.add_argument('cap_size', metavar='Cap Size', type=str, default=cap)
parser.add_argument('discont', metavar='Discontinuity', type=str, default=discontinuity)

args = parser.parse_args(['15', '410'])

discontinuity = args.discont
cap = args.cap_size

def get_Lauren_Pred(cap, precursor):
    file_path = '../../experimental/cross_secs_dat/lauren_pred/' + cap + 'caps_deg.dat'
    if precursor == '410':
        ind = 1
    elif precursor == '660':
        ind = 4
    df = pd.read_csv(file_path, header=None, sep=' ')
    dat_bins = df[0].values
    dat_times = -np.abs(np.asarray([line.split('+/-')[0] for line in df[ind].values], dtype=np.float))
    dat_errors = np.asarray([line.split('+/-')[1] for line in df[ind].values], dtype=np.float)
    return dat_bins, dat_times, dat_errors

def get_Model_Pred(cap, precursor):
    file_path = '../../experimental/cross_secs_dat/model_pred/' + cap + 'caps_deg_results.csv'
    df = pd.read_csv(file_path)
    df = df.loc[:, ['file', precursor+'pred', precursor+'err', precursor+'amp', precursor+'qual']]
    
    return df['file'].values, df[precursor+'pred'].values, df[precursor+'err'].values, df[precursor+'qual'].values

def cap2latlon(bins, units='deg'):
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
        if units == 'deg':
            latlon[i,0] = lat
            latlon[i,1] = lon
        else:
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

lauren_bins, lauren_times, lauren_errors = get_Lauren_Pred(cap, discontinuity)
lauren_latlon = cap2latlon(lauren_bins)

model_bins, model_times, model_errors, model_qual = get_Model_Pred(cap, discontinuity)
model_latlon = cap2latlon(model_bins)
model_bins = model_bins[model_qual > 0.4]
model_latlon = model_latlon[model_qual > 0.4]
model_times = model_times[model_qual > 0.4]
model_errors = model_errors[model_qual > 0.4]

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
cbar.set_ticks(mtick.MultipleLocator(5))
cbar.minorticks_on()
cbar.set_label('Time Before Main Arrival [s]')
for l, lon_0 in enumerate([-180, 0]):
    globe_m = Basemap(projection='moll', lon_0=lon_0, resolution='c', ax=ax[0][l])
    globe_m.drawmapboundary(fill_color='lightgray')
    globe_m.drawcoastlines(color='gray')
    globe_m.drawparallels([-45, 0, 45])
    globe_m.drawmeridians(np.arange(0.,360.,45.))
    #globe.fillcontinents(color='black')
    #globe.scatter(lauren_latlon[85:,0], lauren_latlon[85:,1], color='red', s=10, latlon=True)
    globe_m.scatter(model_latlon[:,1], model_latlon[:,0], c=model_times, s=50,
                    latlon=True, cmap=cmap, norm=norm, zorder=10)
    
    globe_l = Basemap(projection='moll', lon_0=lon_0, resolution='c', ax=ax[2][l])
    globe_l.drawmapboundary(fill_color='lightgray')
    globe_l.drawcoastlines(color='gray')
    globe_l.drawparallels([-45, 0, 45])
    globe_l.drawmeridians(np.arange(0.,360.,45.))
    #globe.fillcontinents(color='black')
    #globe.scatter(lauren_latlon[85:,0], lauren_latlon[85:,1], color='red', s=10, latlon=True)
    globe_l.scatter(lauren_latlon[:,1], lauren_latlon[:,0], c=lauren_times, s=50,
                    latlon=True, cmap=cmap, norm=norm, zorder=10)
    #plt.show()
fig.tight_layout(pad=0.5)

# figs showing time diffs
both_bins = np.intersect1d(lauren_bins, model_bins)
find_match = lambda bins_arr: np.asarray([i for i in range(len(bins_arr)) if bins_arr[i] in both_bins])
model_inds = find_match(model_bins)
lauren_inds = find_match(lauren_bins)

diff_times = np.abs(model_times[model_inds] - lauren_times[lauren_inds])
disagree = diff_times > lauren_errors[lauren_inds]

#print(diff_times[disagree])
#print(model_bins[model_inds][disagree][np.argmax(diff_times[disagree])])
#print('model', model_times[model_inds][disagree][np.argmax(diff_times[disagree])])
#print('lauren', lauren_times[lauren_inds][disagree][np.argmax(diff_times[disagree])],
#      '+/-', lauren_errors[lauren_inds][np.argmax(diff_times[disagree])])
#fig2, ax2 = plt.subplots(nrows=2)

rows2 = 16
cols2 = rows2 + 1
fig2 = plt.figure()
ax2 = [plt.subplot2grid((rows2,cols2), (0,0), colspan=rows2, rowspan=rows2//2, fig=fig2),
      plt.subplot2grid((rows2,cols2), (rows2//2,0), colspan=rows2, rowspan=rows2//2, fig=fig2),
      plt.subplot2grid((rows2,cols2), (0,rows2), colspan=1, rowspan=rows2, fig=fig2)]
fig.text(0.46, 0.925, 'Model Map', fontweight='bold')
fig.text(0.45, 0.375, 'Bootstrap Map', fontweight='bold')
fig.text(0.475, 0.525, 'S'+discontinuity+'S', fontweight='bold')

min_time2, max_time2 = get_MinMax_Times(diff_times, diff_times)
norm2 = mpl.colors.Normalize(vmin=min_time2, vmax=max_time2)
cbar2 = mpl.colorbar.ColorbarBase(ax2[2], cmap=cmap, norm=norm2, orientation='vertical')
cbar2.ax.tick_params(length=5)
cbar2.set_ticks(mtick.MultipleLocator(0.5))
cbar2.minorticks_on()
cbar2.set_label('Absolute Error [s]')
for l, lon_0 in enumerate([0, 180]):
    globe_diff = Basemap(projection='moll', lon_0=lon_0, resolution='c', ax=ax2[l])
    globe_diff.drawmapboundary(fill_color='lightgray')
    globe_diff.drawcoastlines(color='gray')
    globe_diff.drawparallels([-45, 0, 45])
    globe_diff.drawmeridians(np.arange(0.,360.,45.))
    #globe.fillcontinents(color='black')
    #globe.scatter(lauren_latlon[85:,0], lauren_latlon[85:,1], color='red', s=10, latlon=True)
    globe_diff.scatter(model_latlon[:,1][disagree], model_latlon[:,0][disagree],
                       c=diff_times[disagree], s=50, latlon=True, cmap='YlGnBu', zorder=10)
fig2.tight_layout(pad=0.5)
plt.show()
