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
mpl.rcParams['font.size'] = 20
mpl.rcParams['figure.titlesize'] = 'large'
mpl.rcParams['legend.fontsize'] = 'medium'
mpl.rcParams['xtick.major.size'] = 12
mpl.rcParams['xtick.minor.size'] = 8
mpl.rcParams['xtick.labelsize'] = 18
mpl.rcParams['ytick.major.size'] = 12
mpl.rcParams['ytick.minor.size'] = 8
mpl.rcParams['ytick.labelsize'] = 18

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
    arrivals = arrivals.flatten()
    
    arrivals_inds, gcarcs, qualities, errors, amps, arrivals = filter_Data(arrivals != 0, arrivals_inds, gcarcs,
                                                                           qualities, errors, amps, arrivals)
    arrivals_inds, gcarcs, qualities, errors, amps, arrivals = filter_Data(gcarcs >= 100, arrivals_inds, gcarcs,
                                                                           qualities, errors, amps, arrivals)
    arrivals_inds, gcarcs, qualities, errors, amps, arrivals = filter_Data(arrivals > -300, arrivals_inds, gcarcs,
                                                                           qualities, errors, amps, arrivals)
    
    return gcarcs, arrivals, qualities

def discontinuity_model(dat_path, precursor, depth, model):
    df_disc = pd.read_csv('{}S{}Stimes_{}_{}.dat'.format(dat_path, precursor, depth, model), sep=' ', header=None)
    df_main = pd.read_csv('{}SStimes_{}_{}.dat'.format(dat_path, depth, model), sep=' ', header=None)
    df_disc[1] = (df_disc - df_main)[1]
    func = interp1d(df_disc[0].values, df_disc[1].values)
    
    return func

def find_Relevant(arrivals, gcarcs, qualities, prem_model, iasp_model, uncertainty, sigmas):
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
    return gcarcs, arrivals, qualities
    
q1 = 0.6
#q2 = 0.7
gcarcs, arrivals, qualities = get_Predictions('/home/jorgeagr/Documents/seismology/experimental/ss_ind_precursors/',
                            'SS_corrected_preds.csv', qual_cut=q1)
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
prem_660_70 = discontinuity_model('/home/jorgeagr/Documents/seismology/experimental/ss_ind_precursors/phase_times/',
                                '660', '70', 'prem')
iasp_660_0 = discontinuity_model('/home/jorgeagr/Documents/seismology/experimental/ss_ind_precursors/phase_times/',
                                '660', '0', 'iasp')
iasp_660_70 = discontinuity_model('/home/jorgeagr/Documents/seismology/experimental/ss_ind_precursors/phase_times/',
                                '660', '70', 'iasp')

prem_520_0 = discontinuity_model('/home/jorgeagr/Documents/seismology/experimental/ss_ind_precursors/phase_times/',
                                '520', '0', 'prem')
prem_520_70 = discontinuity_model('/home/jorgeagr/Documents/seismology/experimental/ss_ind_precursors/phase_times/',
                                '520', '70', 'prem')
iasp_520_0 = discontinuity_model('/home/jorgeagr/Documents/seismology/experimental/ss_ind_precursors/phase_times/',
                                '520', '0', 'iasp')
iasp_520_70 = discontinuity_model('/home/jorgeagr/Documents/seismology/experimental/ss_ind_precursors/phase_times/',
                                '520', '70', 'iasp')
# can extend this to include the 70 km easily. just have the models be *args in the function and iterate over them
gcarcs410, arrivals410, qualities410 = find_Relevant(arrivals, gcarcs, qualities, prem_410_0, iasp_410_0, 10, 1)
gcarcs660, arrivals660, qualities660 = find_Relevant(arrivals, gcarcs, qualities, prem_660_0, iasp_660_0, 10, 1)
gcarcs520, arrivals520, qualities520 = find_Relevant(arrivals, gcarcs, qualities, prem_520_0, iasp_520_0, 10, 1)

cmin, cmax = qualities.min(), qualities.max()
norm = mpl.colors.Normalize(vmin=cmin, vmax=1)
cmap = mpl.cm.Reds

x = np.linspace(100, 180, num=100)
fig, ax = plt.subplots()
#divider = make_axes_locatable(ax)
#cax = divider.append_axes("right", size="2%", pad=0.05)
#cbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, orientation='vertical')
#cbar.set_label('Quality Factor')

ax.scatter(gcarcs, arrivals, 5, color='black')#5, color='gainsboro')
#ax.scatter(gcarcs410, arrivals410, 10, c=qualities410, cmap='Reds')
#ax.scatter(gcarcs660, arrivals660, 10, c=qualities660, cmap='Reds')
#ax.scatter(gcarcs520, arrivals520, 10, c=qualities520, cmap='Reds')

# 0 Km Event
ax.plot(x, prem_410_0(x), color='red', label='PREM')
ax.plot(x, iasp_410_0(x), '--', color='red', label='IASP')
ax.plot(x, prem_660_0(x), color='red')
ax.plot(x, iasp_660_0(x), '--', color='red')

ax.set_xlim(90, 180)
ax.set_ylim(-100, -300)
ax.yaxis.set_major_locator(mtick.MultipleLocator(50))
ax.yaxis.set_minor_locator(mtick.MultipleLocator(10))
ax.xaxis.set_major_locator(mtick.MultipleLocator(20))
ax.xaxis.set_minor_locator(mtick.MultipleLocator(10))
ax.set_ylabel('SdS-SS (s)')
ax.set_xlabel('Epicentral Distance (deg)')
ax.invert_yaxis()
ax.legend()
#ax.set_title('Qualities {}%-{}%'.format(q1*100, q2*100))
fig.tight_layout(pad=0.5)
fig.savefig('../sds_tdiff_epidist_corrected.png', dpi=200)#_{}_{}.png'.format(q1*100, q2*100))