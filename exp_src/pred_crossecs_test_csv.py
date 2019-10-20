#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 10:34:06 2019

@author: jorgeagr
"""

import numpy as np
import os
import obspy
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from sklearn.cluster import DBSCAN

height = 8#16
width = 12#25

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

def cut_Window(cross_sec, times, t_i, t_f):
    init = np.where(times == np.round(t_i, 1))[0][0]
    end = np.where(times == np.round(t_f, 1))[0][0]
    
    return cross_sec[init:end]

def shift_Max(seis, pred_var):
    data = seis.data
    time = seis.times()
    arrival = 0
    new_arrival = seis.stats.sac[pred_var]
    #for i in range(3):
    while (new_arrival - arrival) != 0:
        arrival = new_arrival
        init = np.where(time > (arrival - 1))[0][0]
        end = np.where(time > (arrival + 1))[0][0]
        
        amp_max = np.argmax(np.abs(data[init:end]))
        time_ind = np.arange(init, end, 1)[amp_max]
        
        new_arrival = time[time_ind]
        #print(new_arr)
        #if np.abs(new_arr - arrival) < 1e-2:
        #    break
    return arrival

# Picked by Lauren
#cs = obspy.read('../../seismograms/cross_secs/5caps_wig/0.087_3.96.sac') # good
#cs = obspy.read('../../seismograms/cross_secs/5caps_wig/0.785_3.74.sac') # meh
#cs = obspy.read('../../seismograms/cross_secs/5caps_wig/0.523_1.17.sac') # bad

# Randomly picked
#cs = obspy.read('../../seismograms/cross_secs/5caps_wig/0.087_0.54.sac')
#cs = obspy.read('../../seismograms/cross_secs/5caps_wig/0.087_2.70.sac')

#cs = obspy.read('../../seismograms/cross_secs/15caps_wig/0.261_2.17.sac')
#cs = obspy.read('../../seismograms/cross_secs/15caps_wig/0_0.sac')
#cs = obspy.read('../../seismograms/cross_secs/15caps_wig/n0.261_3.72.sac')
#cs = obspy.read('../../seismograms/cross_secs/15caps_wig/0.523_5.46.sac')

#cs = obspy.read('../../seismograms/cross_secs/10caps_wig/0.174_0.19.sac')
#cs = obspy.read('../../seismograms/cross_secs/10caps_wig/0.523_4.68.sac')
#cs = obspy.read('../../seismograms/cross_secs/10caps_wig/0_4.08.sac')
#cs = obspy.read('../../seismograms/cross_secs/10caps_wig/n0.872_4.29.sac')

#cs = obspy.read('../../seismograms/cross_secs/10caps_wig/0.174_0.19.sac')
cs = obspy.read('../../seismograms/cross_secs/10caps_wig/0.348_1.76.sac')

file = '0.087_3.96.sac'
with open('5caps_wig_preds.csv') as pred_csv:
    for line in pred_csv:
        if file in line:
            pred_line = line
            break

cs = cs[0].resample(10)
times = cs.times()

shift = -cs.stats.sac.b
b = cs.stats.sac.b + shift
e = cs.stats.sac.e + shift
arr_410 = float(pred_line.split(',')[1]) + shift
arr_660 = float(pred_line.split(',')[3]) + shift

cs_norm = cs.data / np.abs(cs.data).max()
fig, ax = plt.subplots()
ax.plot(times, cs_norm, color='black')
for i, ar in enumerate([arr_660, arr_410]):
    ax.axvline(ar, color='blue', linestyle='--')
    #ax.text(ar-5, 0.1, np.sort(counts_pos)[-2:][i], rotation=90, fontsize=16)
#ax.axvline(ar, color='blue', linestyle='--', label='positive model')
#for i, ar in enumerate(arrivals_neg[np.argsort(counts_neg)][-5:]):
#   ax.axvline(ar, color='red', linestyle='--')
#   ax.text(ar-5, 0.1, np.sort(counts_neg)[-5:][i], rotation=90, fontsize=16)
#ax.axvline(ar, color='red', linestyle='--', label='negative model')
ax.set_ylim(cs_norm.min(), cs_norm.max())
ax.set_xlim(times.min(), times.max())
ax.set_xlabel('Time [s]')
ax.set_ylabel('Amplitude')
ax.set_title('5caps_wig/0.087_3.96')
ax.xaxis.set_minor_locator(mtick.MultipleLocator(10))
#ax.legend(loc='upper right')
fig.tight_layout()