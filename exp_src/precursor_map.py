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
discontinuity = '410'
cap = '5'

parser = argparse.ArgumentParser(description='Model vs Lauren Discontinuity Map')
parser.add_argument('cap_size', metavar='Cap Size', type=str, default=cap)
parser.add_argument('discont', metavar='Discontinuity', type=str, default=discontinuity)

args = parser.parse_args()

discontinuity = args.discont
cap = args.cap_size

# Deprecated
#def get_Lauren_Pred(cap, precursor):
#    file_path = 'cross_secs_dat/lauren_pred/' + cap + 'caps_S' + precursor + 'S.dat'
#    with open(file_path) as datfile:
#        dat_bins = np.asarray([line.split(' ')[0] for line in datfile])
#    dat_times = np.loadtxt(file_path, usecols=(1, 2))
#    return dat_bins, dat_times[:,0], dat_times[:,1]
#
#def get_Lauren_Pred_Bootstraps(cap, precursor):
#    '''
#    As per Lauren:
#    "Columns are: Time, slowness, mean amplitude value, standard deviation
#    So you will use $4 as the error measurement. $3 should correspond to the original cross-section, but may not for messy data."
#    '''
#    if precursor == '410':
#        min_t, max_t = -165, -145
#    elif precursor == '660':
#        min_t, max_t = -235, -215
#    
#    cap_path = 'cross_secs_dat/other_lauren/' + cap + 'caps/'
#    dat_bins = np.sort([file.rstrip('_bootstrap.dat') for file in os.listdir(cap_path)])
#    dat_times = np.zeros(len(dat_bins))
#    dat_errors = np.zeros(len(dat_bins))
#    for i, bin_file in enumerate(dat_bins):
#        bin_file += '_bootstrap.dat'
#        dat_file = np.loadtxt(cap_path + bin_file)
#        ind_range = np.asarray([i for i in range(len(dat_file[:,0])) if min_t < dat_file[i,0] < max_t])
#        arrival_ind = dat_file[ind_range,2].argmax()
#        dat_times[i] = dat_file[ind_range,0][arrival_ind] # arrival time
#        dat_errors[i] = dat_file[ind_range,3][arrival_ind] # error?? of what?
#    
#    return dat_bins, dat_times, dat_errors

def get_Lauren_Pred(cap, precursor):
    file_path = 'cross_secs_dat/lauren_pred/' + cap + 'caps_deg.dat'
    if precursor == '410':
        ind = 1
    elif precursor == '660':
        ind = 4
    df = pd.read_csv(file_path, header=None, sep=' ')
    dat_bins = df[0].values
    dat_times = np.asarray([line.split('+/-')[0] for line in df[ind].values], dtype=np.float)
    dat_errors = np.asarray([line.split('+/-')[1] for line in df[ind].values], dtype=np.float)
    return dat_bins, dat_times, dat_errors

def get_Model_Pred(cap, precursor):
    file_path = 'cross_secs_dat/model_pred/' + cap + 'caps_deg_results.csv'
    df = pd.read_csv(file_path)
    df = df.loc[:, ['file', precursor+'pred', precursor+'err', precursor+'amp', precursor+'qual']]
    
    return df['file'].values, df[precursor+'pred'].values, df[precursor+'err'].values

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

'''
TEMPORARY. REMOVE WHEN LAUREN FIXES. Only needed for the first pred, not bootstraps
'''
#lauren_times = lauren_times-4.2


model_bins, model_times, model_errors = get_Model_Pred(cap, discontinuity)
model_latlon = cap2latlon(model_bins)

'''
TEMPORARY. Figure how to deal with this?? 
Finding extra stuff ruins the file. It will find, say, something before the 410 and the 410.
But since only keep the 2 strongest predictions, 660 may be strong enough but gets thrown away!
Find solution to this...
'''
#less = np.where(-165 < model_times)
#greater = np.where(model_times < -145)
#inds = np.intersect1d(less, greater)

#model_latlon = model_latlon[inds]
#model_times = model_times[inds]



#model_times = np.ones(len(model_times)) * 0
#lauren_times = np.ones(len(lauren_times)) * 0




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
plt.show()
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