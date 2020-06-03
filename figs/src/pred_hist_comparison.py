#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 12:56:03 2020

@author: jorgeagr
"""

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

def get_Predictions_CC(window_size, qual_cut=0.6):
    df = pd.read_csv('/home/jorgeagr/Documents/seismology/experimental/ss_ind_precursors/crosscorr{}.txt'.format(window_size),
                     header=None, sep=' ')
    df = df.fillna(0)
    pred_inds = np.array([1, 4, 7, 10, 13])
    
    gcarcs = np.zeros(len(df))
    df_model = pd.read_csv('/home/jorgeagr/Documents/seismology/experimental/ss_ind_precursors/SS_corrected_preds.csv')
    for i, file in enumerate(df[0].values):
        if file.rstrip('.s_fil') in df_model['file'].values:
            gcarcs[i] = df_model.loc[i,'gcarc']
    
    df = df.loc[gcarcs > 0, :]
    df['gcarcs'] = gcarcs
    df.loc[:, 13] = pd.to_numeric(df[13], errors='coerce')
    
    files = df[0].values
    arrivals = df.values[:,pred_inds].astype(np.float)#.flatten()
    gcarcs = df['gcarcs']
    qualities = df.values[:,pred_inds+2].astype(np.float)
    
    arrivals[qualities < qual_cut] = 0
    qualities = qualities.flatten()
    arrivals_inds = np.meshgrid(np.arange(arrivals.shape[1]),
                               np.arange(arrivals.shape[0]))[1].flatten()
    gcarcs = np.meshgrid(np.arange(arrivals.shape[1]), gcarcs)[1].flatten()
    files = np.meshgrid(np.arange(arrivals.shape[1]), files)[1].flatten()
    arrivals = arrivals.flatten()
    
    arrivals_inds, gcarcs, qualities, arrivals, files = filter_Data(arrivals != 0, arrivals_inds, gcarcs,
                                                                           qualities, arrivals, files)
    arrivals_inds, gcarcs, qualities, arrivals, files = filter_Data(gcarcs >= 100, arrivals_inds, gcarcs,
                                                                           qualities, arrivals, files)
    arrivals_inds, gcarcs, qualities, arrivals, files = filter_Data(arrivals > -300, arrivals_inds, gcarcs,
                                                                           qualities, arrivals, files)
    
    return gcarcs, arrivals, qualities, files

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
  

# 410 models
prem_410_0 = discontinuity_model('/home/jorgeagr/Documents/seismology/experimental/ss_ind_precursors/phase_times/',
                                '410', '0', 'prem')
iasp_410_0 = discontinuity_model('/home/jorgeagr/Documents/seismology/experimental/ss_ind_precursors/phase_times/',
                                '410', '0', 'iasp')
# 660 models
prem_660_0 = discontinuity_model('/home/jorgeagr/Documents/seismology/experimental/ss_ind_precursors/phase_times/',
                                '660', '0', 'prem')
iasp_660_0 = discontinuity_model('/home/jorgeagr/Documents/seismology/experimental/ss_ind_precursors/phase_times/',
                                '660', '0', 'iasp')
x = np.linspace(100, 180, num=100)
fig = plt.figure()
a3 = plt.subplot2grid((8,3), (3,0), colspan=2, rowspan=3, fig=fig)
a4 = plt.subplot2grid((8,3), (3,2), colspan=1, rowspan=3, sharey=a3, fig=fig)
a1 = plt.subplot2grid((8,3), (0,0), colspan=2, rowspan=3, sharex=a3, fig=fig)
a2 = plt.subplot2grid((8,3), (0,2), colspan=1, rowspan=3, sharey=a1, sharex=a4, fig=fig)
a5 = plt.subplot2grid((8,3), (6,0), colspan=3, rowspan=2, fig=fig)  
ax = [[a1, a2],
      [a3, a4],
      a5,]
g_i = 120
g_f = g_i + 10
titles = ('CNN: {} to {} deg / >{} Quality / {} Seismograms', 'CC: {} to {} deg / >{} Quality / {} Seismograms')
for i in [1, 0]:
    if i == 0:
        q1 = 0.6
        #q2 = 0.7
        file = 'corrected'
        gcarcs, arrivals, errors, qualities, amplitudes, files = get_Predictions('/home/jorgeagr/Documents/seismology/experimental/ss_ind_precursors/',
                                    'SS_{}_preds.csv'.format(file), qual_cut=q1)
    else:
        q1 = 0.9379
        gcarcs, arrivals, qualities, files = get_Predictions_CC(20, qual_cut=q1)
        errors = qualities
    
    gcarcs410, arrivals410, errors410, qualities410, files410, model410 = find_Relevant(arrivals, gcarcs, errors, qualities, files,
                                                                              -130, -185, 5, 2) 
    gcarcs660, arrivals660, errors660, qualities660, files660, model660 = find_Relevant(arrivals, gcarcs, errors, qualities, files,
                                                                              -200, -250, 5, 2)
    
    ax[i][0].scatter(gcarcs, arrivals, marker='.', color='black')
    ax[i][0].plot(x, prem_410_0(x), color='red', linewidth=4, label='PREM')
    ax[i][0].plot(x, iasp_410_0(x), '--', color='red', linewidth=4, label='IASP')
    ax[i][0].plot(x, prem_660_0(x), color='red', linewidth=4)
    ax[i][0].plot(x, iasp_660_0(x), '--', color='red', linewidth=4)
    ax[i][0].plot(x, model410.predict(x.reshape(-1,1)).flatten(),
            color='orange', linewidth=4, label='Data Weighted Fit')
    ax[i][0].plot(x, model660.predict(x.reshape(-1,1)).flatten(), color='orange', linewidth=4)
    ax[i][0].set_xlim(100, 180)
    ax[i][0].set_ylim(-100, -300)
    ax[i][0].yaxis.set_major_locator(mtick.MultipleLocator(50))
    ax[i][0].yaxis.set_minor_locator(mtick.MultipleLocator(10))
    ax[i][0].xaxis.set_major_locator(mtick.MultipleLocator(20))
    ax[i][0].xaxis.set_minor_locator(mtick.MultipleLocator(10))
    if i == 0:
        plt.setp(ax[i][0].get_xticklabels(), visible=False)
        plt.setp(ax[i][1].get_xticklabels(), visible=False)
        plt.setp(ax[i][1].get_yticklabels(), visible=False)
    if i == 1:
        ax[i][0].set_ylabel('SdS-SS (s)', fontsize=36)
        ax[i][0].set_xlabel('Epicentral Distance (deg)', fontsize=36)
        ax[i][1].set_xlabel('Frequency', fontsize=36)
        plt.setp(ax[i][1].get_yticklabels(), visible=False)
        ax[i][0].legend()
        ax[i][1].set_title('Picks from {} to {} deg'.format(g_i, g_f))
    ax[i][0].invert_yaxis()
    
    ax[i][1].set_ylim(-300, -100)
    ax[i][1].set_xlim(0, 225)
    seis_total = np.unique(files[(g_i < gcarcs) & (gcarcs < g_f)]).shape[0]
    seis_410 = np.unique(files410[(g_i < gcarcs410) & (gcarcs410 < g_f)]).shape[0]
    seis_660 = np.unique(files660[(g_i < gcarcs660) & (gcarcs660 < g_f)]).shape[0]
    freq, _, _ = ax[i][1].hist(arrivals[(g_i < gcarcs) & (gcarcs < g_f)],
                   np.arange(-300, -100, 1), color='tab:blue', orientation="horizontal")
    freq410, _, _ = ax[i][1].hist(arrivals410[(g_i < gcarcs410) & (gcarcs410 < g_f)],
                      np.arange(-300, -100, 1), color='tab:blue', orientation="horizontal")
    freq660, _, _ = ax[i][1].hist(arrivals660[(g_i < gcarcs660) & (gcarcs660 < g_f)],
                      np.arange(-300, -100, 1), color='tab:blue', orientation="horizontal")
    if i == 0:
        ax[i][1].text(125, -112, 'CNN Quality: >{}'.format(q1), ha='center')
        ax[i][1].text(125, -130, '{} Seismograms'.format(seis_total), ha='center')
        ax[i][1].text(freq410.max()/2, model410.predict(np.array([[g_i + 5]]))[0]+10, '410-km Cluster: {:.2f}'.format(seis_410/seis_total), ha='center')
        ax[i][1].text(freq410.max()/2, model660.predict(np.array([[g_i + 5]]))[0]+10, '660-km Cluster: {:.2f}'.format(seis_660/seis_total), ha='center')
    else:
        ax[i][1].text(125, -112, 'CC Quality: >{}'.format(q1), ha='center')
        ax[i][1].text(125, -130, '{} Seismograms'.format(seis_total), ha='center')
        ax[i][1].text(freq410.max(), model410.predict(np.array([[g_i + 5]]))[0]+10, '410-km Cluster: {:.2f}'.format(seis_410/seis_total), ha='center')
        ax[i][1].text(freq410.max(), model660.predict(np.array([[g_i + 5]]))[0]+10, '660-km Cluster: {:.2f}'.format(seis_660/seis_total), ha='center')
    #ax[i][1].text(model410.predict(np.array([[g_i + 5]]))[0]-10, freq410.max()+0.5, '410-km Cluster: {:.2f}'.format(seis_410/seis_total))
    #ax[i][1].text(model660.predict(np.array([[g_i + 5]]))[0]-10, freq660.max()+0.5, '660-km Cluster: {:.2f}'.format(seis_660/seis_total))
    #ax[i][1].set_title(titles[i].format(g_i, g_f, q1, seis_total))
    ax[i][1].xaxis.set_major_locator(mtick.MultipleLocator(50))
    ax[i][1].xaxis.set_minor_locator(mtick.MultipleLocator(25))
    ax[i][1].yaxis.set_major_locator(mtick.MultipleLocator(50))
    ax[i][1].yaxis.set_minor_locator(mtick.MultipleLocator(10))

df410 = pd.DataFrame(data={'file':files410,
                           'time': arrivals410,
                           'error': errors410,
                           'quality': qualities410})
df660 = pd.DataFrame(data={'file': files660,
                           'time': arrivals660,
                           'error': errors660,
                           'quality': qualities660})

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
init = 1000
cut = 3500
ax[2].axhline(cut/10 + seis.stats.sac.b, color='black', linestyle='--')
ax[2].set_xlim(98, 177)
ax2 = ax[2].twiny()
ax2.set_xlim(98, 177)
ax[2].set_ylim(-300,seis.stats.sac.e)
ax2.set_ylim(-300,seis.stats.sac.e)
for i in range(ep.shape[0]):
    mina = np.abs(amp[i,init:cut]).min()
    maxa = np.abs(amp[i,init:cut]).max()
    ax[2].plot(amp[i,init:cut]/maxa+ep[i,0], t[i,init:cut]+seis.stats.sac.b, color='black')
    for j in range(2):
        x_grid = np.array([ep[i,j] - bin/2, ep[i,j] + bin/2])
        ax[2].fill_between(x_grid, ar[i,j] - er[i,j]*2, ar[i,j] + er[i,j]*2, color='red')
    #ax.axvline(ar[i,0], (ep[i,0]-2.5 - 99)/(176-99), (ep[i,0]+2.5 - 99)/(176-99), color='red')
    #ax.axvline(ar[i,1], ep[i,0]-1, ep[i,0]+1)
    ax2.plot(amp[i,cut:]/np.abs(amp[i,cut:]).max()+ep[i,0], t[i,cut:]+seis.stats.sac.b, color='black')
ax2.axes.get_xaxis().set_visible(False)    
#ax[2].set_ylabel('Time (s)')
ax[2].set_xlabel('Epicentral Distance (deg)', fontsize=36)
ax[2].xaxis.set_major_locator(mtick.MultipleLocator(10))
ax[2].xaxis.set_minor_locator(mtick.MultipleLocator(5))
ax[2].yaxis.set_major_locator(mtick.MultipleLocator(100))
ax[2].yaxis.set_minor_locator(mtick.MultipleLocator(50))
ax[2].invert_xaxis()
ax2.invert_xaxis()

fig.tight_layout(pad=0.5)
fig.savefig('../preds_410_660.pdf'.format(file), dpi=60)
'''    
g_i = 120
g_f = g_i + 10
fig, ax = plt.subplots(ncols=2)
titles = ('CNN: {} to {} deg / >{} Quality / {} Seismograms', 'CC: {} to {} deg / >{} Quality / {} Seismograms')
for i in range(2):
    if i == 0:
        q1 = 0.6
        #q2 = 0.7
        file = 'corrected'
        gcarcs, arrivals, qualities, files = get_Predictions('/home/jorgeagr/Documents/seismology/experimental/ss_ind_precursors/',
                                    'SS_{}_preds.csv'.format(file), qual_cut=q1)
    else:
        q1 = 0.937
        #q2 = 0.7
        gcarcs, arrivals, qualities, files = get_Predictions_CC(20, qual_cut=q1)
    
    gcarcs410, arrivals410, qualities410, files410, model410 = find_Relevant(arrivals, gcarcs, qualities, files,
                                                                              -130, -185, 5, 2) 
    gcarcs660, arrivals660, qualities660, files660, model660 = find_Relevant(arrivals, gcarcs, qualities, files,
                                                                              -200, -250, 5, 2)
    
    ax[i].set_xlim(-300, -100)
    ax[i].set_ylim(0, 225)
    seis_total = np.unique(files[(g_i < gcarcs) & (gcarcs < g_f)]).shape[0]
    seis_410 = np.unique(files410[(g_i < gcarcs410) & (gcarcs410 < g_f)]).shape[0]
    seis_660 = np.unique(files660[(g_i < gcarcs660) & (gcarcs660 < g_f)]).shape[0]
    freq, _, _ = ax[i].hist(arrivals[(g_i < gcarcs) & (gcarcs < g_f)], np.arange(-300, -100, 1), color='tab:blue')
    freq410, _, _ = ax[i].hist(arrivals410[(g_i < gcarcs410) & (gcarcs410 < g_f)], np.arange(-300, -100, 1), color='tab:blue')
    freq660, _, _ = ax[i].hist(arrivals660[(g_i < gcarcs660) & (gcarcs660 < g_f)], np.arange(-300, -100, 1), color='tab:blue')
    ax[i].text(model410.predict(np.array([[g_i + 5]]))[0]-10, freq410.max()+0.5, '410-km Cluster: {:.2f}'.format(seis_410/seis_total))
    ax[i].text(model660.predict(np.array([[g_i + 5]]))[0]-10, freq660.max()+0.5, '660-km Cluster: {:.2f}'.format(seis_660/seis_total))
    ax[i].set_title(titles[i].format(g_i, g_f, q1, seis_total))
    ax[i].yaxis.set_major_locator(mtick.MultipleLocator(50))
    ax[i].yaxis.set_minor_locator(mtick.MultipleLocator(25))
    ax[i].xaxis.set_major_locator(mtick.MultipleLocator(50))
    ax[i].xaxis.set_minor_locator(mtick.MultipleLocator(10))
    ax[i].set_xlabel('SdS-SS (s)')
    ax[i].set_ylabel('Frequency')
fig.tight_layout(pad=0.5)
fig.savefig('../precursor_{}hists.pdf'.format(g_i), dpi=200)
'''