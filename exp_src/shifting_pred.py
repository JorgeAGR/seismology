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


golden_ratio = (np.sqrt(5) + 1) / 2
width = 12
height = width / golden_ratio

mpl.rcParams['figure.figsize'] = (width, height)
mpl.rcParams['font.size'] = 28
mpl.rcParams['figure.titlesize'] = 'large'
mpl.rcParams['legend.fontsize'] = 'small'
mpl.rcParams['xtick.major.size'] = 12
mpl.rcParams['xtick.minor.size'] = 8
mpl.rcParams['xtick.labelsize'] = 24
mpl.rcParams['ytick.major.size'] = 12
mpl.rcParams['ytick.minor.size'] = 8
mpl.rcParams['ytick.labelsize'] = 24
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

files = np.sort(os.listdir('../../seismograms/SS_kept_test/'))

preds = np.zeros(len(files))
actuals = np.zeros(len(files))

for i, file in enumerate(files):
    seis = obspy.read('../../seismograms/SS_kept_test/'+file)[0]
    preds[i] = seis.stats.sac.t7
    actuals[i] = seis.stats.sac.t6 - seis.stats.sac.b

shift_error = preds - actuals
print(len(np.where(shift_error<0)[0])/len(shift_error))
fig, ax = plt.subplots()
weights = np.ones_like(shift_error)/len(shift_error)
shift_hist = ax.hist(shift_error, np.arange(-0.5, 0.5, 0.02), histtype='step', align='mid', 
        color='black', linewidth=2, weights=weights, cumulative=False,)# label='Shifted Windows')
#simple_hist = ax.hist(simple_error, np.arange(-1, 1, 0.1), histtype='step', align='mid',
#               color='green', linewidth=2, weights=weights, cumulative=False, label='Single Window')
ax.set_xlim(-0.5, 0.5)
ax.set_ylim(0, 0.2)
ax.xaxis.set_major_locator(mtick.MultipleLocator(0.1))
ax.yaxis.set_major_locator(mtick.MultipleLocator(0.1))
ax.xaxis.set_minor_locator(mtick.MultipleLocator(0.02))
ax.yaxis.set_minor_locator(mtick.MultipleLocator(0.02))
ax.set_xlabel(r'$t_{pred} - t_{actual}$ [s]')
ax.set_ylabel('Fraction of seismograms')
#ax.legend()
plt.tight_layout()
fig.savefig('../figs/shift_v_simple_hist.png', dpi=250)

fig2, ax2 = plt.subplots()
shift_cum = ax2.hist(np.abs(shift_error), np.arange(0, 1.1, 0.001), histtype='step', align='mid', 
        color='black', linewidth=2, weights=weights, cumulative=True,)# label='Shifted Windows')
#simple_cum = ax2.hist(np.abs(shift_error), np.arange(0, 1.1, 0.001), histtype='step', align='mid', 
#        color='green', linewidth=1.5, weights=weights, cumulative=True, linestyle='--', label='Single Window')
ax2.axhline(0.95, linestyle='--', color='red')
ax2.set_xlim(-0.01, 0.5)
ax2.set_ylim(0, 1.05)
ax2.xaxis.set_major_locator(mtick.MultipleLocator(0.5))
ax2.yaxis.set_major_locator(mtick.MultipleLocator(0.2))
ax2.xaxis.set_minor_locator(mtick.MultipleLocator(0.1))
ax2.yaxis.set_minor_locator(mtick.MultipleLocator(0.05))
ax2.set_xlabel(r'$|t_{pred} - t_{actual}| [s]$')
ax2.set_ylabel('Fraction of counts')
#ax2.legend(loc='lower right')
plt.tight_layout()
fig2.savefig('../figs/shift_v_simple_cumhist.png', dpi=250)
