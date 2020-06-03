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

def get_Model_Pred(dataname, precursor):
    file_path = '../../experimental/ss_ind_precursors/S{}S_{}_results.csv'.format(precursor, dataname)
    df = pd.read_csv(file_path)
    latlon_df = pd.read_csv('../../experimental/ss_ind_precursors/SSbouncepoints.dat', sep=' ', header=None)
    latlon_df.loc[:,0] = [x.rstrip('.s_fil') for x in latlon_df[0].values]
    #latlon_df[0].values[latlon_df.loc[:,0].isin(df['file'].values)]
    latlon_df = latlon_df.loc[latlon_df.loc[:,0].isin(df['file'].values),:]
    # mismatching files, so temp line
    df = df.loc[df.loc[:,'file'].isin(latlon_df[0].values),:]
    
    latlon = latlon_df.values[:,1:]
    #latlon = np.zeros((len(df), 2))
    #latlon[:,0] = df['lat']
    #latlon[:,1] = df['lon']
    return latlon, df['time'].values, df['error'].values, df['quality'].values, df['file'].values
'''
def get_Lauren_Pred(window_size):
    file_path = '../../experimental/ss_ind_precursors/crosscor{}.txt'.format(window_size)
    if precursor == '410':
        ind = 3
    elif precursor == '660':
        ind = 5
    df = pd.read_csv(file_path, header=None, sep=' ')
    dat_bins = df[1].values
    dat_times = df[ind].values
    return dat_bins, dat_times
'''
def get_Lauren_Cap_Pred(cap, precursor):
    file_path = '../../experimental/ss_ind_precursors/picked_vespas/{}deg_correlatedtime.dat'.format(cap)
    if precursor == '410':
        ind = 3
    elif precursor == '660':
        ind = 5
    df = pd.read_csv(file_path, header=None, sep=' ')
    dat_bins = df[1].values
    dat_times = -np.abs(df[ind].values)
    return dat_bins, dat_times

def get_Lauren_Pred(precursor):
    caps = ('15', '10', '7.5', '5')
    dat_bins = np.array([])
    dat_times = np.array([])
    for cap in caps:
        file_path = '../../experimental/ss_ind_precursors/picked_vespas/{}deg_correlatedtime.dat'.format(cap)
        if precursor == '410':
            ind = 3
        elif precursor == '660':
            ind = 5
        df = pd.read_csv(file_path, header=None, sep=' ')
        dat_bins = np.hstack([dat_bins, df[1].values])
        dat_times = np.hstack([dat_times, df[ind].values])
    dat_times = -np.abs(dat_times)
    return dat_bins, dat_times

def get_Bootstrap_Pred(cap, precursor):
    file_path = '../../experimental/cross_secs_dat/lauren_pred/{}caps_deg.dat'.format(cap)
    if precursor == '410':
        ind = 1
    elif precursor == '660':
        ind = 4
    df = pd.read_csv(file_path, header=None, sep=' ')
    dat_bins = df[0].values
    dat_times = np.asarray([line.split('+/-')[0] for line in df[ind].values], dtype=np.float)
    dat_errors = np.asarray([line.split('+/-')[1] for line in df[ind].values], dtype=np.float)
    return dat_bins, dat_times, dat_errors

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

precursor = '410'
dataname = 'corrected'

model_latlon, model_times, model_errors, model_qual, files = get_Model_Pred(dataname, precursor)

lauren_bins, lauren_times = get_Lauren_Pred(precursor)
lauren_latlon = cap2latlon(lauren_bins)

#bootstrap_bins, bootstrap_times, bootstrap_errors = get_Bootstrap_Pred(cap, precursor)
#bootstrap_latlon = cap2latlon(bootstrap_bins)

norm = mpl.colors.Normalize(vmin=model_times.mean()-2*model_times.std(), vmax=model_times.mean()+2*model_times.std())
def plot_Map(latlon, times, title, color_norm = norm, s = 2):
    fig = plt.figure()
    ax = [plt.subplot2grid((15,1), (0,0), colspan=1, rowspan=7, fig=fig),
          plt.subplot2grid((15,1), (7,0), colspan=1, rowspan=7, fig=fig),
          plt.subplot2grid((15,1), (14,0), colspan=1, rowspan=1, fig=fig)]
    #fig.suptitle(title, x=0.25, y=0.6)
    
    #cmap = {'410': mpl.cm.RdYlBu, '660': mpl.cm.RdYlBu_r}[precursor]
    colors = {'410': ['red', 'yellow', 'blue'], '660': ['blue', 'yellow', 'red']}[precursor]
    cmap = mpl.colors.LinearSegmentedColormap.from_list("", colors)
    
    #min_time, max_time = get_MinMax_Times(times, times)
    #norm = mpl.colors.Normalize(vmin=min_time, vmax=max_time)
    #norm = mpl.colors.Normalize(vmin=times.mean()-2*times.std(), vmax=times.mean()+2*times.std())
    cbar = mpl.colorbar.ColorbarBase(ax[2], cmap=cmap, norm=color_norm, orientation='horizontal')
    cbar.ax.tick_params(length=5)
    cbar.set_ticks(mtick.MultipleLocator(5))
    cbar.minorticks_on()
    cbar.set_label('S{}S-SS (s)'.format(precursor))
    for l, lon_0 in enumerate([-180, 0]):
        globe_m = Basemap(projection='moll', lon_0=lon_0, resolution='c', ax=ax[l])
        globe_m.drawmapboundary(fill_color='white')#'gainsboro')
        globe_m.drawcoastlines(color='gray')
        globe_m.drawparallels([-45, 0, 45])
        globe_m.drawmeridians(np.arange(0.,360.,45.))
        #globe_m.pcolormesh(model_latlon[:,1], model_latlon[:,0], times, latlon=True, cmap=cmap)
        globe_m.scatter(latlon[:,1], latlon[:,0], c=times, s=s,
                        latlon=True, cmap=cmap)#, norm=norm, zorder=10)
        #plt.show()
    fig.tight_layout(pad=0.5)
    return fig

def plot_MapComparison(latlon_top, times_top, latlon_bot, times_bot, color_norm = norm, s1=2, s2=15):
    fig = plt.figure()
    ax = [plt.subplot2grid((15,2), (0,0), colspan=1, rowspan=7, fig=fig),
          plt.subplot2grid((15,2), (0,1), colspan=1, rowspan=7, fig=fig),
          plt.subplot2grid((15,2), (7,0), colspan=2, rowspan=1, fig=fig),
          plt.subplot2grid((15,2), (8,0), colspan=1, rowspan=7, fig=fig),
          plt.subplot2grid((15,2), (8,1), colspan=1, rowspan=7, fig=fig)]
    
    #cmap = {'410': mpl.cm.RdYlBu, '660': mpl.cm.RdYlBu_r}[precursor]
    colors = {'410': ['red', 'yellow', 'blue'], '660': ['blue', 'yellow', 'red']}[precursor]
    cmap = mpl.colors.LinearSegmentedColormap.from_list("", colors)
    
    #min_time, max_time = get_MinMax_Times(times, times)
    #norm = mpl.colors.Normalize(vmin=min_time, vmax=max_time)
    #norm = mpl.colors.Normalize(vmin=times.mean()-2*times.std(), vmax=times.mean()+2*times.std())
    cbar = mpl.colorbar.ColorbarBase(ax[2], cmap=cmap, norm=color_norm, orientation='horizontal')
    cbar.ax.tick_params(length=5)
    cbar.set_ticks(mtick.MultipleLocator(5))
    cbar.minorticks_on()
    cbar.set_label('S{}S-SS (s)'.format(precursor))
    latlon = (latlon_top, latlon_bot)
    times = (times_top, times_bot)
    s = (s1, s2)
    for i in range(2):
        for l, lon_0 in enumerate([-180, 0]):
            globe_m = Basemap(projection='moll', lon_0=lon_0, resolution='c', ax=ax[l+3*i])
            globe_m.drawmapboundary(fill_color='white')#'gainsboro')
            globe_m.drawcoastlines(color='black')
            globe_m.drawparallels([-45, 0, 45])
            globe_m.drawmeridians(np.arange(0.,360.,45.))
            #globe_m.pcolormesh(model_latlon[:,1], modehl_latlon[:,0], times, latlon=True, cmap=cmap)
            globe_m.scatter(latlon[i][:,1], latlon[i][:,0], c=times[i], s=s[i],
                            latlon=True, cmap=cmap, norm=norm, zorder=10)
            #plt.show()
        fig.tight_layout(pad=0.5)
    return fig

def plot_Both(dataname):
    d410_latlon, d410_times, d410_errors, d410_qual, files = get_Model_Pred(dataname, 410)
    d660_latlon, d660_times, d660_errors, d660_qual, files = get_Model_Pred(dataname, 660)
    
    fig = plt.figure()
    ax = [plt.subplot2grid((15,2), (0,0), colspan=1, rowspan=7, fig=fig),
          plt.subplot2grid((15,2), (7,0), colspan=1, rowspan=7, fig=fig),
          plt.subplot2grid((15,2), (0,1), colspan=1, rowspan=7, fig=fig),
          plt.subplot2grid((15,2), (7,1), colspan=1, rowspan=7, fig=fig),
          plt.subplot2grid((15,2), (14,0), colspan=1, rowspan=1, fig=fig),
          plt.subplot2grid((15,2), (14,1), colspan=1, rowspan=1, fig=fig)]
    latlon = (d410_latlon, d660_latlon)
    times = (d410_times, d660_times)
    for i, precursor in enumerate(('410','660')):
        color_norm = mpl.colors.Normalize(vmin=times[i].mean()-2*times[i].std(), vmax=times[i].mean()+2*times[i].std())
        colors = {'410': ['red', 'yellow', 'blue'], '660': ['blue', 'yellow', 'red']}[precursor]
        cmap = mpl.colors.LinearSegmentedColormap.from_list("", colors)
        
        cbar = mpl.colorbar.ColorbarBase(ax[-2+i], cmap=cmap, norm=color_norm, orientation='horizontal')
        cbar.ax.tick_params(length=5)
        cbar.set_ticks(mtick.MultipleLocator(5))
        cbar.minorticks_on()
        cbar.set_label('S{}S-SS (s)'.format(precursor))
        for l, lon_0 in enumerate([-180, 0]):
            globe_m = Basemap(projection='moll', lon_0=lon_0, resolution='c', ax=ax[l+2*i])
            globe_m.drawmapboundary(fill_color='white')#'gainsboro')
            globe_m.drawcoastlines(color='black')
            globe_m.drawparallels([-45, 0, 45])
            globe_m.drawmeridians(np.arange(0.,360.,45.))
            #globe_m.pcolormesh(model_latlon[:,1], modehl_latlon[:,0], times, latlon=True, cmap=cmap)
            globe_m.scatter(latlon[i][:,1], latlon[i][:,0], c=times[i], s=2,
                            latlon=True, cmap=cmap, norm=color_norm, zorder=10)
            #plt.show()
        fig.tight_layout(pad=0.5)
    
    return fig

def plot_Diff(dataname):
    file_path = '../../experimental/ss_ind_precursors/SBothS_{}_results.csv'.format(dataname)
    df = pd.read_csv(file_path)
    latlon_df = pd.read_csv('../../experimental/ss_ind_precursors/SSbouncepoints.dat', sep=' ', header=None)
    latlon_df.loc[:,0] = [x.rstrip('.s_fil') for x in latlon_df[0].values]
    #latlon_df[0].values[latlon_df.loc[:,0].isin(df['file'].values)]
    latlon_df = latlon_df.loc[latlon_df.loc[:,0].isin(df['file'].values),:]
    # mismatching files, so temp line
    df = df.loc[df.loc[:,'file'].isin(latlon_df[0].values),:]
    
    latlon = latlon_df.values[:,1:]
    
    times = df['660time'].values - df['410time'].values
    
    fig = plt.figure(figsize=(15,7))
#    ax = [plt.subplot2grid((15,1), (0,0), colspan=1, rowspan=7, fig=fig),
#          plt.subplot2grid((15,1), (7,0), colspan=1, rowspan=7, fig=fig),
#          plt.subplot2grid((15,1), (14,0), colspan=1, rowspan=1, fig=fig)]
    ax = [plt.subplot2grid((15,2), (0,0), colspan=1, rowspan=14, fig=fig),
          plt.subplot2grid((15,2), (0,1), colspan=1, rowspan=14, fig=fig),
          plt.subplot2grid((15,2), (14,0), colspan=2, rowspan=1, fig=fig),]
    
    #cmap = {'410': mpl.cm.RdYlBu, '660': mpl.cm.RdYlBu_r}[precursor]
    colors = ['blue', 'yellow', 'red']
    cmap = mpl.colors.LinearSegmentedColormap.from_list("", colors)
    
    #min_time, max_time = get_MinMax_Times(times, times)
    #norm = mpl.colors.Normalize(vmin=min_time, vmax=max_time)
    color_norm = mpl.colors.Normalize(vmin=times.mean()-2*times.std(), vmax=times.mean()+2*times.std())
    cbar = mpl.colorbar.ColorbarBase(ax[2], cmap=cmap, norm=color_norm, orientation='horizontal')
    cbar.ax.tick_params(length=5)
    #cbar.set_ticks(mtick.MultipleLocator(5))
    cbar.minorticks_on()
    cbar.set_label('S660S-S410S (s)'.format(precursor))
    for l, lon_0 in enumerate([-180, 0]):
        globe_m = Basemap(projection='moll', lon_0=lon_0, resolution='c', ax=ax[l])
        globe_m.drawmapboundary(fill_color='white')#'gainsboro')
        globe_m.drawcoastlines(color='gray')
        globe_m.drawparallels([-45, 0, 45])
        globe_m.drawmeridians(np.arange(0.,360.,45.))
        #globe_m.pcolormesh(model_latlon[:,1], model_latlon[:,0], times, latlon=True, cmap=cmap)
        globe_m.scatter(latlon[:,1], latlon[:,0], c=times, s=2,
                        latlon=True, cmap=cmap)#, norm=norm, zorder=10)
        #plt.show()
    fig.tight_layout(pad=0.5)
    return fig

#model_fig = plot_Map(model_latlon, model_times, '{} Real Data Map'.format(precursor))
#model_fig.savefig('../s{}s_{}_map.png'.format(precursor, dataname), dpi=500)
#bootnorm = mpl.colors.Normalize(vmin=bootstrap_times.mean()-2*bootstrap_times.std(), vmax=bootstrap_times.mean()+2*bootstrap_times.std())
#bootstrap_fig = plot_Map(bootstrap_latlon, bootstrap_times, '{} Boostrap Map'.format(precursor), s=5)
#bootstrap_fig.savefig('../s{}s_bootstrap_{}cap_map.png'.format(precursor, cap), dpi=500)
#set_norm = {'410': mpl.colors.Normalize(vmin=-173, vmax=-143), '660': mpl.colors.Normalize(vmin=-245, vmax=-215)}[precursor]
#comparison_fig = plot_MapComparison(model_latlon, model_times, lauren_latlon, lauren_times)#, color_norm=set_norm)
#comparison_fig.savefig('../s{}s_{}_data_vs_cap_map.eps'.format(precursor, dataname), dpi=200)
both_fig = plot_Both(dataname)
both_fig.savefig('../{}_map.pdf'.format(dataname), dpi=60)

diff_fig = plot_Diff(dataname)
diff_fig.savefig('../{}_diff_map.pdf'.format(dataname), dpi=60)

'''    
fig = plt.figure()
ax = [plt.subplot2grid((15,1), (0,0), colspan=1, rowspan=7, fig=fig),
      plt.subplot2grid((15,1), (7,0), colspan=1, rowspan=7, fig=fig),
      plt.subplot2grid((15,1), (14,0), colspan=1, rowspan=1, fig=fig)]

cmap = mpl.cm.YlGnBu
min_time, max_time = get_MinMax_Times(model_times, model_times)
#norm = mpl.colors.Normalize(vmin=min_time, vmax=max_time)
norm = mpl.colors.Normalize(vmin=model_times.mean()-2*model_times.std(), vmax=model_times.mean()+2*model_times.std())
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
fig.savefig('../s{}s_realdata_map.png'.format(precursor), dpi=500)
#plt.close()
'''
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