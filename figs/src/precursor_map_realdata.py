#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 23:26:18 2020

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

def get_Model_Pred(precursor):
    file_path = 'ss_ind_precursors/S{}S_individual_results.csv'.format(precursor)
    df = pd.read_csv(file_path)
    latlon_df = pd.read_csv('ss_ind_precursors/SSbouncepoints.dat', sep=' ', header=None)
    latlon_df.loc[:,0] = [x.rstrip('.s_fil') for x in latlon_df[0].values]
    #latlon_df[0].values[latlon_df.loc[:,0].isin(df['file'].values)]
    latlon_df = latlon_df.loc[latlon_df.loc[:,0].isin(df['file'].values),:]
    # mismatching files, so temp line
    df = df.loc[df.loc[:,'file'].isin(latlon_df[0].values),:]
    
    latlon = latlon_df.values[:,1:]
    #latlon = np.zeros((len(df), 2))
    #latlon[:,0] = df['lat']
    #latlon[:,1] = df['lon']
    keys = df.keys()[3:]
    return latlon, df[keys[0]].values, df[keys[1]].values, df[keys[3]].values, df['file'].values

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

precursor = '660'
model_latlon, model_times, model_errors, model_qual, files = get_Model_Pred(precursor)

#fig, ax = plt.subplots(nrows=3)
fig = plt.figure()
ax = [plt.subplot2grid((15,1), (0,0), colspan=1, rowspan=7, fig=fig),
      plt.subplot2grid((15,1), (7,0), colspan=1, rowspan=7, fig=fig),
      plt.subplot2grid((15,1), (14,0), colspan=1, rowspan=1, fig=fig)]

cmap = mpl.cm.YlGnBu
min_time, max_time = get_MinMax_Times(model_times, model_times)
norm = mpl.colors.Normalize(vmin=min_time, vmax=max_time)
cbar = mpl.colorbar.ColorbarBase(ax[2], cmap=cmap, norm=norm, orientation='horizontal')
cbar.ax.tick_params(length=5)
cbar.set_ticks(mtick.MultipleLocator(1))
#cbar.minorticks_on()
cbar.set_label('S{}S-SS (s)'.format(precursor))
for l, lon_0 in enumerate([-180, 0]):
    globe_m = Basemap(projection='moll', lon_0=lon_0, resolution='c', ax=ax[l])
    globe_m.drawmapboundary(fill_color='lightgray')
    globe_m.drawcoastlines(color='gray')
    globe_m.drawparallels([-45, 0, 45])
    globe_m.drawmeridians(np.arange(0.,360.,45.))
    #globe_m.pcolormesh(model_latlon[:,1], model_latlon[:,0], model_times, latlon=True, cmap=cmap)
    globe_m.scatter(model_latlon[:,1], model_latlon[:,0], c=model_times, s=10,
                    latlon=True, cmap=cmap)#, norm=norm, zorder=10)
    #plt.show()
fig.tight_layout(pad=0.5)
fig.savefig('/home/jorgeagr/Downloads/s{}s_realdata_map.png'.format(precursor), dpi=500)
#plt.close()

import obspy
from mpl_toolkits.axes_grid1 import make_axes_locatable

def get_GCARC_TimeDiff(precursor):
    # Lazy copy-paste implementation of stuff
    df = pd.read_csv('ss_ind_precursors/S{}S_individual_results.csv'.format(precursor))    
    gcarc = np.zeros(len(df['file'].values))
    time = df['{}pred'.format(precursor)].values
    for f, file in enumerate(df['file'].values):
        seis = obspy.read(file_path + file + '.s_fil')[0]
        # I am aware this is just model_times since model_times is time diff
        # between precursor and main arrival, seismos are aligned to 0.
        # just future proofing in case.
        time[f] = time[f] - seis.stats.sac.t6
        gcarc[f] = seis.stats.sac.gcarc
    return time, gcarc, df['{}qual'.format(precursor)].values


file_path = '/home/jorgeagr/Documents/seismograms/SS_kept/'
time410, gcarc410, qual410 = get_GCARC_TimeDiff('410')
time660, gcarc660, qual660 = get_GCARC_TimeDiff('660')
qualconcat = np.concatenate([qual410, qual660], axis=0)
cmin, cmax = qualconcat.min(), qualconcat.max()
norm = mpl.colors.Normalize(vmin=cmin, vmax=cmax)

fig, ax = plt.subplots()
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="2%", pad=0.05)
cmap = mpl.cm.Reds
ax.scatter(gcarc410, time410, 10, c=qual410, cmap=cmap)
ax.scatter(gcarc660, time660, 10, c=qual660, cmap=cmap)
cbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, orientation='vertical')
cbar.set_label('Quality Factor')
ax.yaxis.set_major_locator(mtick.MultipleLocator(50))
ax.yaxis.set_minor_locator(mtick.MultipleLocator(10))
ax.xaxis.set_major_locator(mtick.MultipleLocator(20))
ax.xaxis.set_minor_locator(mtick.MultipleLocator(10))
ax.set_ylabel('SdS-SS (s)')
ax.set_xlabel('Epicentral Distance (deg)')
ax.set_ylim(0, -400)
ax.invert_yaxis()
#ax.set_xlim(90, 180)
fig.tight_layout(pad=0.5)
fig.savefig('/home/jorgeagr/Downloads/sds_tdiff_epidist.png'.format(precursor), dpi=250)
plt.close()

'''
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
'''