#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 14:33:03 2020

@author: jorgeagr
"""
import numpy as np
import os
import obspy
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from sklearn.cluster import DBSCAN

height = 16
width = 25

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

model = 'SS40'
pos_model = load_model('../../seyesmolite/models/{}/{}.h5'.format(model, model))
time_window = 40

cap = '15'
file = 'n15_156.50'
cs = obspy.read('../../../seismograms/cross_secs/'+cap+'caps_deg/'+file+'.sac')

cs = cs[0].resample(10)
times = cs.times()

shift = -cs.stats.sac.b
begin_time = cs.stats.sac.b#-np.abs(-280) # Seconds before main arrival. Will become an input.
begin_time = np.round(begin_time + shift, decimals=1)
end_time = cs.stats.sac.e#-np.abs(-80) # ditto above
end_time = np.round(end_time + shift, decimals=1)

time_i_grid = np.arange(begin_time, end_time - time_window, 0.1)
time_f_grid = np.arange(begin_time + time_window, end_time, 0.1)

window_preds = np.zeros(len(time_i_grid))
for i, t_i, t_f in zip(range(len(time_i_grid)), time_i_grid, time_f_grid):
    if t_f > shift:
        pass#break
    cs_window = cut_Window(cs, times, t_i, t_f)
    cs_window = cs_window / np.abs(cs_window).max()
    # Take the absolute value of the prediction to remove any wonky behavior in finding the max
    # Doesn't matter since they are bad predictions anyways
    cs.stats.sac.t6 = np.abs(pos_model.predict(cs_window.reshape(1, len(cs_window), 1))[0][0]) + t_i
    window_preds[i] += cs.stats.sac.t6#shift_Max(cs, 't6')

eps=0.05
min_neighbors=2
dbscan = DBSCAN(eps, min_neighbors)
dbscan.fit(window_preds.reshape(-1,1))
clusters, counts = np.unique(dbscan.labels_, return_counts=True)

grouped_preds = window_preds
for c in clusters:
    if c != -1:
        grouped_preds[np.where(dbscan.labels_ == c)] = np.ones_like(grouped_preds[dbscan.labels_ == c]) * grouped_preds[dbscan.labels_ == c].mean()

if -1 in clusters:
    clusters = clusters[1:]
    counts = counts[1:]
arrivals = np.zeros(len(clusters))
arrivals_qual = np.zeros(len(clusters))
for c in clusters:
    arrivals[c] = np.mean(window_preds[dbscan.labels_ ==  c])
    arrivals_qual[c] = counts[c]/400
times = times-shift

fig, ax = plt.subplots(nrows=2, sharex=True)
cut = 2500
ax[0].plot(times[:cut], cs.data[:cut] / np.abs(cs.data[:cut]).max(), color='black')
for i, ar in enumerate(arrivals[np.argsort(arrivals_qual)][-4:]):
    ax[0].axvline(ar-shift, color='red', linestyle='--')
    ax[0].text(ar-5-shift, 0.2, np.sort(arrivals_qual)[-4:][i], rotation=90, fontsize=16)
ax[0].axvline(ar-shift, color='red', linestyle='--', label='model')
ax[0].axvline(cut/10 - shift, color='black', linestyle='--')

ax2 = ax[0].twinx()
ax2.plot(times[cut:], cs.data[cut:] / np.abs(cs.data[cut:]).max(), color='black')
ax2.axes.get_yaxis().set_visible(False)

ax[0].set_ylim(-1, 1)
ax[0].set_xlim(times.min(), times.max())
ax[1].set_xlabel('Time [s]', fontsize=36)
ax[0].xaxis.set_major_locator(plt.NullLocator())
ax[0].set_ylabel('Relative Amplitude', fontsize=36)

ax[1].hist(grouped_preds-shift, np.arange(begin_time, end_time+0.3, 0.3)-shift, color='red',
           weights=np.ones(len(grouped_preds)) / (time_window*10), linewidth=2)
ax[1].set_ylabel('Quality', fontsize=36)
ax[1].set_ylim(0, 1)
ax[1].xaxis.set_major_locator(mtick.MultipleLocator(50))
ax[1].xaxis.set_minor_locator(mtick.MultipleLocator(25))
ax[0].yaxis.set_major_locator(mtick.MultipleLocator(1))
ax[0].yaxis.set_minor_locator(mtick.MultipleLocator(0.5))
ax[1].yaxis.set_major_locator(mtick.MultipleLocator(0.2))
ax[1].yaxis.set_minor_locator(mtick.MultipleLocator(0.1))
fig.tight_layout(pad=1)
fig.subplots_adjust(wspace=0, hspace=0.1)
fig.savefig('../crosssection_preds.pdf', dpi=200)