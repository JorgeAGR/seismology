#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 13:43:16 2019

@author: jorgeagr
"""
import os
import obspy
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import obspy
from tensorflow.keras.models import load_model

golden_ratio = (np.sqrt(5) + 1) / 2
width = 12
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
'''
shifting_npz = np.load('../train_data_shift_pred.npz')
files = shifting_npz['files'].astype(np.str)
pred_avg = shifting_npz['pred_avg']
pred_err = shifting_npz['pred_err']
flipped = shifting_npz['flipped']

simple_npz = np.load('../train_data_simple_pred.npz')
simple_files = simple_npz['files'].astype(np.str)
simple_pred = simple_npz['pred'].flatten()

actuals_npz = np.load('./train_data/etc/train_data_arrivals.npz')
actuals_files = actuals_npz['files'].astype(np.str)
actuals_indeces = np.argsort(actuals_files)
arrivals = actuals_npz['arrivals'][actuals_indeces]

shift_error = pred_avg - arrivals
simple_error = simple_pred - arrivals
'''
'''
files = np.sort([file for file in os.listdir('/home/jorgeagr/Documents/seismograms/SS_picked/') if '.sac' in file])
#cutoff = 1000
#files = files[:cutoff]
preds = np.zeros(len(files))
actuals = np.zeros(len(files))

for i, file in enumerate(files):
    seis = obspy.read('/home/jorgeagr/Documents/seismograms/SS_picked/'+file)[0]
    preds[i] = seis.stats.sac.t2 - seis.stats.sac.b
    actuals[i] = seis.stats.sac.t6 - seis.stats.sac.b
'''

model = load_model('../../seyesmolite/models/SS40/SS40.h5')
test_files = np.array([file.rstrip('.npz') for file in np.load('../../seyesmolite/models/SS40/train_logs/train_test_split1.npz')['test']])

arrival_var = 't6'
th_arrival_var = 't2'
window_before = 0
window_after = 40
sample_rate = 10
total_time = 400

arrivals = np.zeros(len(test_files))
preds = np.zeros(len(test_files))
for i, file in enumerate(test_files):
    seis = obspy.read('/home/jorgeagr/Documents/seismograms/SS_kept/'+file)[0]
    #preds[i] = seis.stats.sac.t2 - seis.stats.sac.b
    #actuals[i] = seis.stats.sac.t6 - seis.stats.sac.b
    b = seis.stats.sac['b']
    # The beginning time may not be 0, so shift all attribtues to be so
    shift = -b
    b = b + shift
    # End time
    e = seis.stats.sac['e'] + shift
    # Theoretical onset arrival time + shift
    if th_arrival_var == arrival_var:
        th_arrival = seis.stats.sac[arrival_var] + shift - np.random.rand() * 20
    else:
        th_arrival = seis.stats.sac[th_arrival_var] + shift
    # Picked maximum arrival time + shift
    arrival = seis.stats.sac[arrival_var] + shift
    
    # Theoretical arrival may be something unruly, so assign some random
    # shift from the picked arrival
    if not (b < th_arrival < e):
        th_arrival = arrival - 20 * np.random.rand()
    
    amp = seis.data
    time = seis.times()
    
    rand_arrival = th_arrival
    init = int(np.round((rand_arrival - window_before)*sample_rate))
    end = init + total_time
    if not (time[init] < arrival < time[end]):
        init = int(np.round((arrival - 15 * np.random.rand() - window_before)*sample_rate))
        end = init + total_time
    amp_i = amp[init:end]
    # Normalize by absolute peak, [-1, 1]
    amp_i = amp_i / np.abs(amp_i).max()
    arrivals[i] = arrival - time[init]
    preds[i] = model.predict(amp_i.reshape((1, 400, 1)))



shift_error = preds - arrivals
print('avg abs err:', np.abs(shift_error).mean(), '+/-', np.abs(shift_error).std())
print('min error:', np.abs(shift_error).min())
print('max error:', np.abs(shift_error).max())
#print(len(np.where(shift_error<0)[0])/len(shift_error))
fig, ax = plt.subplots(ncols=2, figsize=(15, 8))
weights = np.ones_like(shift_error)/len(shift_error)
shift_hist = ax[0].hist(shift_error, np.arange(-0.5, 0.5, 0.05), histtype='step', align='mid', 
        color='black', linewidth=2)#, weights=weights, cumulative=False,)# label='Shifted Windows')
#simple_hist = ax.hist(simple_error, np.arange(-1, 1, 0.1), histtype='step', align='mid',
#               color='green', linewidth=2, weights=weights, cumulative=False, label='Single Window')
ax[0].set_xlim(-0.5, 0.5)
ax[0].set_ylim(0, 1000)
ax[0].yaxis.set_major_locator(mtick.MultipleLocator(250))
ax[0].yaxis.set_minor_locator(mtick.MultipleLocator(50))
ax[0].xaxis.set_major_locator(mtick.MultipleLocator(0.5))
ax[0].xaxis.set_minor_locator(mtick.MultipleLocator(0.1))
ax[0].set_xlabel(r'$t_{pred} - t_{actual}$ (s)', fontsize=36)
ax[0].set_ylabel('Seismograms', fontsize=36)
#ax.legend()
#fig.tight_layout(pad=0.5)
#fig.savefig('../figs/shift_v_simple_hist.png', dpi=250)
#fig.savefig('../figs/shift_v_simple_hist.svg', dpi=250)

#fig2, ax2 = plt.subplots()
shift_cum = ax[1].hist(np.abs(shift_error), np.arange(0, 1.1, 0.001), histtype='step', align='mid',
        color='black', linewidth=2, weights=weights, cumulative=True,)# label='Shifted Windows')
#simple_cum = ax2.hist(np.abs(shift_error), np.arange(0, 1.1, 0.001), histtype='step', align='mid', 
#        color='green', linewidth=1.5, weights=weights, cumulative=True, linestyle='--', label='Single Window')
ax[1].axhline(0.95, linestyle='--', color='red')
ax[1].set_xlim(-0.01, 0.5)
ax[1].set_ylim(0, 1.05)
ax[1].xaxis.set_major_locator(mtick.MultipleLocator(0.25))
ax[1].yaxis.set_major_locator(mtick.MultipleLocator(0.25))
ax[1].xaxis.set_minor_locator(mtick.MultipleLocator(0.05))
ax[1].yaxis.set_minor_locator(mtick.MultipleLocator(0.05))
ax[1].set_xlabel(r'$|t_{pred} - t_{actual}|$ (s)', fontsize=36)
#ax[1].set_ylabel('Fraction of counts')
#ax2.legend(loc='lower right')
fig.tight_layout(pad=0.5)
fig.savefig('../model_accuracy.pdf', dpi=200)
#fig2.savefig('../figs/shift_v_simple_cumhist.png', dpi=250)
#fig2.savefig('../figs/shift_v_simple_cumhist.svg', dpi=250)
