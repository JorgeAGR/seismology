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
        dat_bins = [line.split(' ')[0] for line in datfile]
    dat_times = np.loadtxt(file_path, usecols=(1, 2))
    return dat_bins, dat_times[:,0], dat_times[:,1]

def get_Model_Pred(cap, precursor):
    return

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

globe = Basemap(projection='moll', lon_0=0, resolution='c')
globe.drawcoastlines()
#globe.fillcontinents(color='black')
#globe.scatter(lauren_latlon[85:,0], lauren_latlon[85:,1], color='red', s=10, latlon=True)

globe.scatter(lauren_latlon[:,1], lauren_latlon[:,0], c=lauren_times, s=20, latlon=True)
plt.show()