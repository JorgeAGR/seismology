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
width = 12
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

lauren_bins, lauren_times, lauren_errors = get_Lauren_Pred('15', '410')
lauren_latlon = cap2latlon(lauren_bins)

model_bins, model_times, model_errors = get_Model_Pred('15', '410')
model_latlon = cap2latlon(model_bins)

fig, ax = plt.subplots(nrows=2, ncols=2)
for l, lon_0 in enumerate([0, 180]):
    globe_m = Basemap(projection='moll', lon_0=lon_0, resolution='c', ax=ax[0][l])
    globe_m.drawmapboundary(fill_color='lightgray')
    globe_m.drawcoastlines(color='gray')
    globe_m.drawparallels([-45, 0, 45])
    globe_m.drawmeridians(np.arange(0.,360.,45.))
    #globe.fillcontinents(color='black')
    #globe.scatter(lauren_latlon[85:,0], lauren_latlon[85:,1], color='red', s=10, latlon=True)
    globe_m.scatter(model_latlon[:,1], model_latlon[:,0], c=model_times, s=50, latlon=True, cmap='YlGnBu', zorder=10)
    
    globe_l = Basemap(projection='moll', lon_0=lon_0, resolution='c', ax=ax[1][l])
    globe_l.drawmapboundary(fill_color='lightgray')
    globe_l.drawcoastlines(color='gray')
    globe_l.drawparallels([-45, 0, 45])
    globe_l.drawmeridians(np.arange(0.,360.,45.))
    #globe.fillcontinents(color='black')
    #globe.scatter(lauren_latlon[85:,0], lauren_latlon[85:,1], color='red', s=10, latlon=True)
    globe_l.scatter(lauren_latlon[:,1], lauren_latlon[:,0], c=lauren_times, s=50, latlon=True, cmap='YlGnBu', zorder=10)
    #plt.show()
    
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