#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 14:43:40 2018

@author: jorgeagr
"""
# Chem 111
# Non-casio calc
# HJLC 225 

import obspy
import numpy as np
from scipy.signal import lombscargle
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['font.size'] = 12
mpl.rcParams['figure.titlesize'] = 'large'
mpl.rcParams['legend.fontsize'] = 'small'

def frequency_grid(time, samples, oversampling):
    period = time[-1] - time[0]
    dt = period / samples
    df = 1/period/oversampling
    fmax = 1/(2*dt)
    fmin = df
    freqs = np.arange(fmin, fmax, df)
    omegas = 2*np.pi*freqs
    return freqs, omegas

# Initial testing
'''
test = obspy.read('20150327.BAK.BHT.s_fil') # Picked
test2 = obspy.read('20150327.146B.BHT.s_fil') # Not picked?

amp = test[0].data

time = np.arange(0, test[0].stats.npts * test[0].stats.delta, test[0].stats.delta) # seconds
s = test[0].stats.sac['t2'] - test[0].stats.sac['b']
scs = test[0].stats.sac['t3'] - test[0].stats.sac['b']

s2 = test2[0].stats.sac['t2'] - test2[0].stats.sac['b']
scs2 = test2[0].stats.sac['t3'] - test2[0].stats.sac['b']
try:
    t6 = test[0].stats.sac['t6'] - test[0].stats.sac['b']
    t6_time = time[np.where(np.isclose(t6, time))]
    t6_amp = amp[np.where(np.isclose(t6, time))]
except:
    pass

fig, ax = plt.subplots(nrows=2, ncols=1)
ax[0].plot(time, amp, linewidth = 1)
ax[1].plot(t6_time, t6_amp, 'or')
ax[0].vlines([s, scs], ymin = min(amp), ymax = max(amp))
ax[0].set_xlim(0, 6000)
ax[0].ticklabel_format(style = 'sci', scilimits = (0,0), axis = 'y', useMathText = True)
ax[0].ticklabel_format(style = 'sci', scilimits = (0,0), axis = 'x', useMathText = True)

ax[1].plot(time, amp, linewidth = 1)
ax[1].set_xlim(s-30, scs)
ax[1].ticklabel_format(style = 'sci', scilimits = (0,0), axis = 'y', useMathText = True)
ax[1].ticklabel_format(style = 'sci', scilimits = (0,0), axis = 'x', useMathText = True)
'''
# Plot 5 seismograms
window = 200

files = []
with open('done.txt') as file:
    for line in file:
        files.append(line.rstrip('\n'))
fig, ax = plt.subplots(nrows=5, ncols=2)
fig.suptitle('Overview of picked seismograms: Ecuador', weight='bold')
plt.subplots_adjust(hspace = 0.4)
for f in enumerate(files):
    ind = f[0]
    ind2 = 0
    file = f[1]
    seis = obspy.read(file)
    amp = seis[0].data
    time = np.arange(0, seis[0].stats.npts * seis[0].stats.delta, seis[0].stats.delta)
    s = seis[0].stats.sac['t2'] - seis[0].stats.sac['b']
    scs = seis[0].stats.sac['t3'] - seis[0].stats.sac['b']
    
    start = np.where(time > s-window)[0][0]
    end = np.where(time < s+window)[0][-1]
    
    t6 = seis[0].stats.sac['t6'] - seis[0].stats.sac['b']
    t6_time = time[np.where(np.isclose(t6, time))]
    #t6_amp = amp[np.where(np.isclose(t6, time))]
    #print(ind, s, t6_time)
    if ind in range(5,10):
        ind = ind - 5
        ind2 += 1
    ax[ind][ind2].set_title(seis[0].stats.station)
    ax[ind][ind2].plot(time[start:end], amp[start:end], linewidth = 1)
    ax[ind][ind2].vlines([s, scs], ymin = min(amp), ymax = max(amp))
    ax[ind][ind2].vlines(t6, ymin = min(amp), ymax = max(amp), color='red')
    ax[ind][ind2].set_xlim(s-window, s+window)
    ax[ind][ind2].ticklabel_format(style = 'sci', scilimits = (0,0), axis = 'y', useMathText = True)
    #ax[ind][ind2].ticklabel_format(style = 'sci', scilimits = (0,0), axis = 'x', useMathText = True)
    if ind2 == 1 and (ind + 5) == 9:
        break

# Array centered around S and ScS
'''
s_ind = np.where(np.isclose(s, time))[0][0]
scs_ind = np.where(np.isclose(scs, time))[0][0]
t = time[s_ind-1200:scs_ind+1]
a = amp[s_ind-1200:scs_ind+1]

fig, ax = plt.subplots(nrows=2, ncols=1)
ax[0].plot(t, a, '.')
ax[0].set_xlim(t[0], t[-1])
'''