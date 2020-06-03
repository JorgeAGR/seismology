#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 17:06:47 2020

@author: jorgeagr
"""
import os
import argparse
import obspy
import numpy as np
import pandas as pd
#import tensorflow.keras.losses
#import tensorflow.keras.metrics
#from tensorflow.losses import huber_loss
from sklearn.cluster import DBSCAN
from sklearn.linear_model import LinearRegression
from scipy.interpolate import interp1d
from time import time as clock
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Settings for plots
golden_ratio = (np.sqrt(5) + 1) / 2
width = 25
height = width / golden_ratio

mpl.rcParams['figure.figsize'] = (width, height)
mpl.rcParams['font.size'] = 22
mpl.rcParams['figure.titlesize'] = 'large'
mpl.rcParams['legend.fontsize'] = 'medium'
mpl.rcParams['xtick.major.size'] = 16
mpl.rcParams['xtick.major.width'] = 2
mpl.rcParams['xtick.minor.width'] = 2
mpl.rcParams['xtick.minor.size'] = 12
mpl.rcParams['xtick.labelsize'] = 36
mpl.rcParams['ytick.major.size'] = 16
mpl.rcParams['ytick.major.width'] = 2
mpl.rcParams['ytick.minor.size'] = 12
mpl.rcParams['ytick.minor.width'] = 2
mpl.rcParams['ytick.labelsize'] = 36
mpl.rcParams['axes.linewidth'] = 2

def filter_Data(condition_array, *arrays):
    new_arrays = []
    for array in arrays:
        new_arrays.append(array[condition_array])
    return new_arrays

def get_Predictions(pred_csv_path, file, qual_cut=0.6):
    df = pd.read_csv(pred_csv_path + file)
    preds = 0 
    for key in df.keys():
        if 'pred' in key:
            preds += 1
    pred_inds = np.array([2+4*i for i in range(preds)])
    err_inds = pred_inds + 1
    amp_inds = err_inds + 1
    qual_inds = amp_inds + 1
    
    files = df['file'].values
    arrivals = df.values[:,pred_inds]#.flatten()
    gcarcs = df['gcarc'].values
    errors = df.values[:,err_inds].flatten()
    amps = df.values[:,amp_inds].flatten()
    qualities = df.values[:,qual_inds]
    
    arrivals[qualities < qual_cut] = 0
    qualities = qualities.flatten()
    arrivals_inds = np.meshgrid(np.arange(arrivals.shape[1]),
                               np.arange(arrivals.shape[0]))[1].flatten()
    gcarcs = np.meshgrid(np.arange(arrivals.shape[1]), gcarcs)[1].flatten()
    files = np.meshgrid(np.arange(arrivals.shape[1]), files)[1].flatten()
    arrivals = arrivals.flatten()
    
    arrivals_inds, gcarcs, qualities, errors, amps, arrivals, files = filter_Data(arrivals != 0, arrivals_inds, gcarcs,
                                                                           qualities, errors, amps, arrivals, files)
    arrivals_inds, gcarcs, qualities, errors, amps, arrivals, files = filter_Data(gcarcs >= 100, arrivals_inds, gcarcs,
                                                                           qualities, errors, amps, arrivals, files)
    arrivals_inds, gcarcs, qualities, errors, amps, arrivals, files = filter_Data(arrivals > -300, arrivals_inds, gcarcs,
                                                                           qualities, errors, amps, arrivals, files)
    
    return gcarcs, arrivals, errors, qualities, amps, files

def discontinuity_model(dat_path, precursor, depth, model):
    df_disc = pd.read_csv('{}S{}Stimes_{}_{}.dat'.format(dat_path, precursor, depth, model), sep=' ', header=None)
    df_main = pd.read_csv('{}SStimes_{}_{}.dat'.format(dat_path, depth, model), sep=' ', header=None)
    df_disc[1] = (df_disc - df_main)[1]
    func = interp1d(df_disc[0].values, df_disc[1].values)
    
    return func

def find_Relevant(arrivals, gcarcs, errors, qualities, files, init_time, end_time, uncertainty, sigmas, weighted=True):
    condition = (arrivals < init_time) & (arrivals > end_time)
    arrivals = arrivals[condition]
    gcarcs = gcarcs[condition]
    errors = errors[condition]
    qualities = qualities[condition]
    files = files[condition]
    
    percent_data = 1
    eps = 5
    clusters = []
    clustering_data = np.vstack([arrivals, gcarcs]).T
    while len(clusters) < 1:
        percent_data += -0.05
        dbscan = DBSCAN(eps=eps, min_samples=len(arrivals)*percent_data)#np.round(len(df)*percent_data))
        #dbscan.fit(arrivals.reshape(-1,1))
        dbscan.fit(clustering_data)
        clusters = np.unique(dbscan.labels_)
        if -1 in clusters:
            clusters = clusters[1:]
    '''
    gcarcs_n = gcarcs
    arrivals_n = arrivals
    errors_n = errors
    qualities_n = qualities
    files_n = files
    old_total = 0
    new_total = len(arrivals)
    init_fit = 0
    while old_total - new_total != 0:
        old_total = len(arrivals)
        if init_fit==0:
            linear = LinearRegression()
            linear.fit(gcarcs[dbscan.labels_==0].reshape(-1,1),
                       arrivals[dbscan.labels_==0], sample_weight=(qualities[dbscan.labels_==0])**weighted)
            arrive_linear = linear.predict(gcarcs.reshape(-1,1)).flatten()
            init_fit = 1
        else:
            linear = LinearRegression()
            linear.fit(gcarcs_n.reshape(-1,1),
                       arrivals_n, sample_weight=(qualities_n)**weighted)
            arrive_linear = linear.predict(gcarcs.reshape(-1,1)).flatten()
        
        zscore = (arrivals - arrive_linear) / uncertainty
        condition = np.abs(zscore) < sigmas
        gcarcs_n = gcarcs[condition]
        arrivals_n = arrivals[condition]
        errors_n = errors[condition]
        qualities_n = qualities[condition]
        files_n = files[condition]
        new_total = len(arrivals)
    '''
    linear = LinearRegression()
    linear.fit(gcarcs[dbscan.labels_==0].reshape(-1,1),
               arrivals[dbscan.labels_==0], sample_weight=(qualities[dbscan.labels_==0])**weighted)
    arrive_linear = linear.predict(gcarcs.reshape(-1,1)).flatten()
    
    zscore = (arrivals - arrive_linear) / uncertainty
    condition = np.abs(zscore) < sigmas
    gcarcs = gcarcs[condition]
    arrivals = arrivals[condition]
    errors = errors[condition]
    qualities = qualities[condition]
    files = files[condition]
    
    linear = LinearRegression()
    linear.fit(gcarcs.reshape(-1,1),
               arrivals, sample_weight=qualities**weighted)
    #return gcarcs_n, arrivals_n, errors_n, qualities_n, files_n, linear
    return gcarcs, arrivals, errors, qualities, files, linear

def find_Relevant_OLD(arrivals, gcarcs, qualities, prem_model, iasp_model, uncertainty, sigmas):
    theory = (np.vstack([prem_model(gcarcs), iasp_model(gcarcs)]))
    inds = np.array([])
    for i in range(theory.shape[0]):
        zscore = (arrivals - theory[i, :]) / uncertainty
        inds = np.concatenate([inds, np.argwhere(np.abs(zscore) < sigmas).flatten()])
    #zscore = (arrivals - theory) / uncertainty
    condition = np.asarray(np.unique(inds), dtype=np.int)#np.abs(zscore) < sigmas
    gcarcs = gcarcs[condition]
    arrivals = arrivals[condition]
    qualities = qualities[condition]
    return gcarcs, arrivals, qualities, files
    
q1 = 0.6
#q2 = 0.7
file = 'corrected'
gcarcs, arrivals, errors, qualities, amplitudes, files = get_Predictions('/home/jorgeagr/Documents/seismology/experimental/ss_ind_precursors/',
                            'SS_{}_preds.csv'.format(file), qual_cut=q1)
#gcarcs, arrivals, qualities = filter_Data(qualities <= q2, gcarcs, arrivals, qualities)

# 410 models
prem_410_0 = discontinuity_model('/home/jorgeagr/Documents/seismology/experimental/ss_ind_precursors/phase_times/',
                                '410', '0', 'prem')
prem_410_70 = discontinuity_model('/home/jorgeagr/Documents/seismology/experimental/ss_ind_precursors/phase_times/',
                                '410', '70', 'prem')
iasp_410_0 = discontinuity_model('/home/jorgeagr/Documents/seismology/experimental/ss_ind_precursors/phase_times/',
                                '410', '0', 'iasp')
iasp_410_70 = discontinuity_model('/home/jorgeagr/Documents/seismology/experimental/ss_ind_precursors/phase_times/',
                                '410', '70', 'iasp')
# 660 models
prem_660_0 = discontinuity_model('/home/jorgeagr/Documents/seismology/experimental/ss_ind_precursors/phase_times/',
                                '660', '0', 'prem')
iasp_660_0 = discontinuity_model('/home/jorgeagr/Documents/seismology/experimental/ss_ind_precursors/phase_times/',
                                '660', '0', 'iasp')

prem_520_0 = discontinuity_model('/home/jorgeagr/Documents/seismology/experimental/ss_ind_precursors/phase_times/',
                                '520', '0', 'prem')
iasp_520_0 = discontinuity_model('/home/jorgeagr/Documents/seismology/experimental/ss_ind_precursors/phase_times/',
                                '520', '0', 'iasp')

# Weighted
gcarcs410, arrivals410, errors410, qualities410, files410, model410 = find_Relevant(arrivals, gcarcs, errors, qualities, files,
                                                                          -130, -185, 5, 2) 
gcarcs660, arrivals660, errors660, qualities660, files660, model660 = find_Relevant(arrivals, gcarcs, errors, qualities, files,
                                                                          -200, -250, 5, 2)
#gcarcs520, arrivals520, qualities520 = find_Relevant(arrivals, gcarcs, qualities, prem_520_0, iasp_520_0, 10, 1)

# Unweighted
#_, _, _, model410_uw = find_Relevant(arrivals, gcarcs, qualities, -130, -185, 5, 2, weighted=False)
#_, _, _, model660_uw = find_Relevant(arrivals, gcarcs, qualities, -200, -250, 5, 2, weighted=False)

x = np.linspace(100, 180, num=100)
fig, ax = plt.subplots()
color = 0
if color:
    grad = amplitudes
    grad_name = 'Amplitude'
    cmin, cmax = grad.min(), grad.max()
    norm = mpl.colors.Normalize(vmin=cmin, vmax=cmax)
    cmap = mpl.cm.hot_r
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.05)
    cbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, orientation='vertical')
    cbar.set_label(grad_name)
    ax.scatter(gcarcs[np.argsort(grad)], arrivals[np.argsort(grad)], marker='.', c=grad, cmap=cmap, norm=norm)
else:
    ax.scatter(gcarcs, arrivals, marker='.', color='black')#5, color='gainsboro')

# Theory Models
if not color:
    ax.plot(x, prem_410_0(x), color='red', linewidth=4, label='PREM')
    ax.plot(x, iasp_410_0(x), '--', color='red', linewidth=4, label='IASP')
    ax.plot(x, prem_660_0(x), color='red', linewidth=4)
    ax.plot(x, iasp_660_0(x), '--', color='red', linewidth=4)
    
    # Data Model
    ax.plot(x, model410.predict(x.reshape(-1,1)).flatten(),
            color='orange', linewidth=4, label='Data Weighted Fit')
    ax.plot(x, model660.predict(x.reshape(-1,1)).flatten(), color='orange', linewidth=4)

# Unweighted models
#ax.plot(x, model410_uw.predict(x.reshape(-1,1)).flatten(),
#        color='green', linewidth=2, label='Data Unweighted Fit')
#ax.plot(x, model660_uw.predict(x.reshape(-1,1)).flatten(), color='green', linewidth=2)

ax.set_xlim(100, 180)
ax.set_ylim(-100, -300)
ax.yaxis.set_major_locator(mtick.MultipleLocator(50))
ax.yaxis.set_minor_locator(mtick.MultipleLocator(10))
ax.xaxis.set_major_locator(mtick.MultipleLocator(20))
ax.xaxis.set_minor_locator(mtick.MultipleLocator(10))
ax.set_ylabel('SdS-SS (s)', fontsize=36)
ax.set_xlabel('Epicentral Distance (deg)', fontsize=36)
ax.invert_yaxis()
if not color:
    ax.legend()
#ax.set_title('Qualities {}%-{}%'.format(q1*100, q2*100))
fig.tight_layout(pad=0.5)
if color:
    fig.savefig('../{}_preds_{}.pdf'.format(file, grad_name), dpi=60)
else:
    fig.savefig('../{}_preds.pdf'.format(file), dpi=60)

df410 = pd.DataFrame(data={'file':files410,
                           'time': arrivals410,
                           'error': errors410,
                           'quality': qualities410})
#df410.to_csv('../../experimental/ss_ind_precursors/S410S_{}_results.csv'.format(file), index=False)
df660 = pd.DataFrame(data={'file': files660,
                           'time': arrivals660,
                           'error': errors660,
                           'quality': qualities660})
#df660.to_csv('../../experimental/ss_ind_precursors/S660S_{}_results.csv'.format(file), index=False)
'''
g_i = 120
g_f = g_i + 10
seis_total = np.unique(files[(g_i < gcarcs) & (gcarcs < g_f)]).shape[0]
seis_410 = np.unique(files410[(g_i < gcarcs410) & (gcarcs410 < g_f)]).shape[0]
seis_660 = np.unique(files660[(g_i < gcarcs660) & (gcarcs660 < g_f)]).shape[0]
fig, ax = plt.subplots()
ax.set_xlim(-300, -100)
freq, _, _ = ax.hist(arrivals[(g_i < gcarcs) & (gcarcs < g_f)], np.arange(-300, -100, 1), color='tab:blue')
freq410, _, _ = ax.hist(arrivals410[(g_i < gcarcs410) & (gcarcs410 < g_f)], np.arange(-300, -100, 1), color='tab:blue')
freq660, _, _ = ax.hist(arrivals660[(g_i < gcarcs660) & (gcarcs660 < g_f)], np.arange(-300, -100, 1), color='tab:blue')
ax.text(model410.predict(np.array([[g_i + 5]]))[0]-10, freq410.max()+0.5, '410-km Cluster: {:.2f}'.format(seis_410/seis_total))
ax.text(model660.predict(np.array([[g_i + 5]]))[0]-10, freq660.max()+0.5, '660-km Cluster: {:.2f}'.format(seis_660/seis_total))
ax.set_title('CNN: {} to {} deg / >{} Quality / {} Seismograms'.format(g_i, g_f, q1, seis_total))
#ax.yaxis.set_major_locator(mtick.MultipleLocator(50))
#ax.yaxis.set_minor_locator(mtick.MultipleLocator(10))
ax.xaxis.set_major_locator(mtick.MultipleLocator(50))
ax.xaxis.set_minor_locator(mtick.MultipleLocator(10))
ax.set_xlabel('SdS-SS (s)')
ax.set_ylabel('Frequency')
fig.tight_layout(pad=0.5)
fig.savefig('../precursor_{}_{}hist_{}.pdf'.format(file, g_i, q1), dpi=200)
'''
'''
dfboth = pd.merge(df410, df660, how='inner', on=['file'])
bin = 2.5
t, ep = np.meshgrid(np.arange(0, 497.3, 0.1), np.arange(100, 180-bin, bin))
amp = np.zeros(t.shape)
ar = np.zeros((len(amp), 2))
er = np.zeros(ar.shape)
for i in range(ep.shape[0]):
    q = 0.9
    while (amp[i] == np.zeros_like(len(amp[i]))).sum() == len(amp[i]):
        for f in dfboth[((dfboth['quality_x'] > q) & (dfboth['quality_y'] > q))].sort_values('quality_y', ascending=False)['file'].values:
            if ep[i,0] < gcarcs[files == f][0] < ep[i,0]+bin:
                seis = obspy.read('/home/jorgeagr/Documents/seismograms/SS_corrected/{}.s_fil'.format(f))[0].resample(10)
                print(gcarcs[files == f][0], dfboth[dfboth.file == f].quality_x.values[0], dfboth[dfboth.file == f].quality_y.values[0])
                amp[i] = seis.data[:4973]
                ar[i,0] = dfboth[dfboth.file == f].time_x.values[0]
                ar[i,1] = dfboth[dfboth.file == f].time_y.values[0]
                er[i,0] = dfboth[dfboth.file == f].error_x.values[0]
                er[i,1] = dfboth[dfboth.file == f].error_y.values[0]
                break
        q += -0.05
        if q <= 0.6:
            break
fig, ax = plt.subplots(figsize=(7,15))
init = 1000
cut = 3500
ax.axvline(cut/10 + seis.stats.sac.b, color='black', linestyle='--')
ax.set_ylim(98, 177)
ax2 = ax.twinx()
ax2.set_ylim(98, 177)
ax.set_xlim(-300,seis.stats.sac.e)
ax2.set_xlim(-300,seis.stats.sac.e)
for i in range(ep.shape[0]):
    mina = np.abs(amp[i,init:cut]).min()
    maxa = np.abs(amp[i,init:cut]).max()
    ax.plot(t[i,init:cut]+seis.stats.sac.b, amp[i,init:cut]/maxa+ep[i,0], color='black')
    for j in range(2):
        y_grid = np.array([ep[i,j] - bin/2, ep[i,j] + bin/2])
        ax.fill_betweenx(y_grid, ar[i,j] - er[i,j]*2, ar[i,j] + er[i,j]*2, color='red')
    #ax.axvline(ar[i,0], (ep[i,0]-2.5 - 99)/(176-99), (ep[i,0]+2.5 - 99)/(176-99), color='red')
    #ax.axvline(ar[i,1], ep[i,0]-1, ep[i,0]+1)
    ax2.plot(t[i,cut:]+seis.stats.sac.b, amp[i,cut:]/np.abs(amp[i,cut:]).max()+ep[i,0], color='black')
ax2.axes.get_yaxis().set_visible(False)    
ax.set_xlabel('Time (s)')
ax.set_ylabel('Epicentral Distance (deg)')
ax.yaxis.set_major_locator(mtick.MultipleLocator(10))
ax.yaxis.set_minor_locator(mtick.MultipleLocator(5))
ax.xaxis.set_major_locator(mtick.MultipleLocator(100))
ax.xaxis.set_minor_locator(mtick.MultipleLocator(50))
fig.tight_layout(pad=0.5)
fig.savefig('../seis_corrected_preds.pdf', dpi=200)
'''