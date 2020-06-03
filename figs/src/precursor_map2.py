#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 14:31:13 2020

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

def get_Model_Bin_Pred(cap, precursor):
    file_path = '../../experimental/ss_ind_precursors/S{}S_{}cap_results.csv'.format(precursor, cap)
    df = pd.read_csv(file_path)
    
    return df['file'].values, df['time'].values, df['error'].values, df['quality'].values
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
def get_Lauren_Bin_Pred(cap, precursor):
    file_path = '../../experimental/ss_ind_precursors/picked_vespas/{}deg_correlatedtime.dat'.format(cap)
    if precursor == '410':
        ind = 3
    elif precursor == '660':
        ind = 5
    df = pd.read_csv(file_path, header=None, sep=' ')
    dat_bins = df[1].values
    dat_times = -np.abs(df[ind].values)
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
cap = 15

model_bins, model_times, model_errors, model_qual = get_Model_Bin_Pred(cap, precursor)
model_latlon = cap2latlon(model_bins)

lauren_bins, lauren_times = get_Lauren_Bin_Pred(cap, precursor)
lauren_latlon = cap2latlon(lauren_bins)

bootstrap_bins, bootstrap_times, bootstrap_errors = get_Bootstrap_Pred(cap, precursor)
bootstrap_latlon = cap2latlon(bootstrap_bins)

'''
# Pearson Correlation Coefficient or Cosine Similarity
m_sort = np.argsort(model_bins)
l_sort = np.argsort(lauren_bins)
both = np.intersect1d(model_bins, lauren_bins)
mod_ind = np.array([i for i in range(len(model_bins)) if model_bins[m_sort][i] in both])
lau_ind = np.array([i for i in range(len(lauren_bins)) if lauren_bins[l_sort][i] in both])
pcc = np.corrcoef(lauren_times[l_sort][lau_ind], model_times[m_sort][mod_ind])
np.dot(lauren_times[l_sort][lau_ind], model_times[m_sort][mod_ind]) / (np.linalg.norm(lauren_times[l_sort][lau_ind])*np.linalg.norm(model_times[m_sort][mod_ind]))
'''

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
                        latlon=True, cmap=cmap, norm=norm, zorder=10)
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
    #cbar.set_ticks(mtick.MultipleLocator(5))
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

def plot_Diff(cap):
    file_410 = '../../experimental/ss_ind_precursors/S410S_{}cap_results.csv'.format(cap)
    file_660 = '../../experimental/ss_ind_precursors/S660S_{}cap_results.csv'.format(cap)
    df410 = pd.read_csv(file_410)
    df660 = pd.read_csv(file_660)
    df = pd.merge(df410, df660, how='inner', on=['file'])
    
#    latlon_df = pd.read_csv('../../experimental/ss_ind_precursors/SSbouncepoints.dat', sep=' ', header=None)
#    latlon_df.loc[:,0] = [x.rstrip('.s_fil') for x in latlon_df[0].values]
#    #latlon_df[0].values[latlon_df.loc[:,0].isin(df['file'].values)]
#    latlon_df = latlon_df.loc[latlon_df.loc[:,0].isin(df['file'].values),:]
#    # mismatching files, so temp line
#    df = df.loc[df.loc[:,'file'].isin(latlon_df[0].values),:]
    
    latlon = cap2latlon(df.file.values)
    
    times = df['time_y'].values - df['time_x'].values
    
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
        globe_m.scatter(latlon[:,1], latlon[:,0], c=times, s=15,
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
comparison_fig = plot_MapComparison(model_latlon, model_times, lauren_latlon, lauren_times, s1=15)#, color_norm=set_norm)
comparison_fig.savefig('../s{}s_{}cap_map.pdf'.format(precursor, cap), dpi=200)

diff_fig = plot_Diff(cap)
diff_fig.savefig('../sdiffs_{}cap_map.pdf'.format(cap), dpi=200)