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
from keras.models import load_model
import keras.losses
import keras.metrics
from tensorflow.losses import huber_loss
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from sklearn.cluster import DBSCAN

height = 16
width = 25

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

keras.losses.huber_loss = huber_loss
pos_model = load_model('../auto_seismo/models/arrival_SS_pos_model_0040.h5')
neg_model = load_model('../auto_seismo/models/arrival_SS_neg_model_0040.h5')
time_window = 40

# Picked by Lauren
#cs = obspy.read('../../seismograms/cross_secs/5caps_wig/0.087_3.96.sac') # good
#cs = obspy.read('../../seismograms/cross_secs/5caps_wig/0.785_3.74.sac') # meh
#cs = obspy.read('../../seismograms/cross_secs/5caps_wig/0.523_1.17.sac') # bad

# Randomly picked
#cs = obspy.read('../../seismograms/cross_secs/5caps_wig/0.087_0.54.sac')
#cs = obspy.read('../../seismograms/cross_secs/5caps_wig/0.087_2.70.sac')

cap = '15'
file = 'n15_156.50'#'n45_232.87'
#cs = obspy.read('../../seismograms/cross_secs/'+cap+'caps_deg/'+file+'.sac')
cs = obspy.read('../../seismograms/SS_kept/19880704A.peru.INU.BHT.s_fil')

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
window_shifted = np.zeros(len(time_i_grid))
for i, t_i, t_f in zip(range(len(time_i_grid)), time_i_grid, time_f_grid):
    if t_f > shift:
        pass#break
    cs_window = cut_Window(cs, times, t_i, t_f)
    cs_window = cs_window / np.abs(cs_window).max()
    # Take the absolute value of the prediction to remove any wonky behavior in finding the max
    # Doesn't matter since they are bad predictions anyways
    cs.stats.sac.t6 = np.abs(pos_model.predict(cs_window.reshape(1, len(cs_window), 1))[0][0]) + t_i
    window_preds[i] += cs.stats.sac.t6#shift_Max(cs, 't6')
    #window_shifted[i] += shift_Max(cs, 't6')
    #break

window_negs = np.zeros(len(time_i_grid))
for i, t_i, t_f in zip(range(len(time_i_grid)), time_i_grid, time_f_grid):
    if t_f > shift:
        pass
    cs_window = -cut_Window(cs, times, t_i, t_f)
    cs_window = cs_window / np.abs(cs_window).max()
    # Take the absolute value of the prediction to remove any wonky behavior in finding the max
    # Doesn't matter since they are bad predictions anyways
    cs.stats.sac.t6 = np.abs(pos_model.predict(cs_window.reshape(1, len(cs_window), 1))[0][0]) + t_i
    window_negs[i] += cs.stats.sac.t6#shift_Max(cs, 't6')
    #break

#arrivals_pos, counts_pos = np.unique(np.round(window_preds, 0), return_counts=True)
#arrivals_neg, counts_neg = np.unique(np.round(window_negs, 0), return_counts=True)
#plateus = np.where(np.diff(np.round(window_preds,1)) == 0)[0] + 1
#arrivals, counts = arrivals[counts > 50], counts[counts > 50]

# Can think of max distance eps as the max allowed variance??
# DBSCAN looks for dense clusters, 
#while True:
def cluster_preds(predictions, eps=0.05, min_neighbors=2):
    dbscan = DBSCAN(eps, min_neighbors)
    dbscan.fit(predictions.reshape(-1,1))
    clusters, counts = np.unique(dbscan.labels_, return_counts=True)
    if -1 in clusters:
        clusters = clusters[1:]
        counts = counts[1:]
    arrivals = np.zeros(len(clusters))
    arrivals_qual = np.zeros(len(clusters))
    for c in clusters:
        arrivals[c] = np.mean(predictions[dbscan.labels_ ==  c])
        arrivals_qual[c] = counts[c]/400
    return arrivals, arrivals_qual

relevant_preds = 8
arrivals_pos, arrivals_pos_qual = cluster_preds(window_preds)
arrivals_neg, arrivals_neg_qual = cluster_preds(window_negs)

# 1
# arrivals_pos[np.argsort(arrivals_pos_qual)][-relevant_preds:]
# [np.argsort(arrivals_pos_qual)][-relevant_preds:]
arrivals_pos_qual = arrivals_pos_qual[arrivals_pos < 2500+shift]
arrivals_neg_qual = arrivals_neg_qual[arrivals_neg < 2500+shift]
arrivals_pos = arrivals_pos[arrivals_pos < 2500+shift]
arrivals_neg = arrivals_neg[arrivals_neg < 2500+shift]

arrivals_pos = arrivals_pos[np.argsort(arrivals_pos_qual)][-relevant_preds:]
arrivals_neg = arrivals_neg[np.argsort(arrivals_neg_qual)][-relevant_preds:]
arrivals_pos_qual = arrivals_pos_qual[np.argsort(arrivals_pos_qual)][-relevant_preds:]
arrivals_neg_qual = arrivals_neg_qual[np.argsort(arrivals_neg_qual)][-relevant_preds:]

# 2
'''
arrivals = []
arrivals_qual = [] 
for i, ar_p in enumerate(arrivals_pos):
    for j, ar_n in enumerate(arrivals_neg):
        if np.abs(ar_p - ar_n) <= 40:
            if (ar_p < ar_n) and (ar_n not in arrivals):
                arrivals.append(ar_n)
                arrivals_qual.append(arrivals_neg_qual[i])
            elif (ar_p > ar_n) and (ar_p not in arrivals):
                arrivals.append(ar_p)
                arrivals_qual.append(arrivals_neg_qual[i])
        else:
            arrivals.append(ar_n)
            arrivals_qual.append(arrivals_neg_qual[i])
'''
keep_pos = []
keep_neg = []
bad_neg = []
for i, ar in enumerate(arrivals_pos):
    dist = np.abs(ar - arrivals_neg)
    close = np.where(dist <= 40)[0]
    if close.shape[0] == 0:
        keep_pos.append(i)
    else:
        if (ar - arrivals_neg[close[0]]) < 0:
            keep_neg.append(close[0])
        else:
            keep_pos.append(i)
            bad_neg.append(close[0])
for j in range(len(arrivals_neg)):
    if (j not in keep_neg) and (j not in bad_neg):
        keep_neg.append(j)

arrivals_pos = arrivals_pos[keep_pos]
arrivals_pos_qual = arrivals_pos_qual[keep_pos]
arrivals_neg = arrivals_neg[keep_neg]
arrivals_neg_qual = arrivals_neg_qual[keep_neg]

arrivals = np.hstack([arrivals_pos, arrivals_neg])
arrivals_qual = np.hstack([arrivals_pos_qual, arrivals_neg_qual])

'''
# Difference between predictions at t+1 and t
preds_diff = np.diff(window_preds)
# Which ones are close 0, to find plateus
zeros = np.isclose(preds_diff, 0, atol=0.05)
# Calculate the difference in indeces between the zeros.
# Where jumps == 1, means they belong in the same plateu
# Anything larger, that is a jump
pred_ind = np.arange(0, len(preds_diff), 1)
pred_ind[~zeros] = 0
jumps = np.zeros(len(preds_diff))
jumps[1:] = np.abs(np.diff(pred_ind))
jumps[jumps == 1] = 1
jumps[jumps > 1] = 0

fig, ax = plt.subplots()
ax.plot(time_i_grid[1:], preds_diff)
ax.plot(time_i_grid[1:][zeros], preds_diff[zeros], '.')
ax.plot(time_i_grid[1:], jumps)
ax.set_ylim(-5, 10)
'''
times = cs.times() - shift
fig, ax = plt.subplots()
ax.plot(time_i_grid-shift, window_preds-shift, '.', color='black')
#for cluster in clusters:#[sorted_ind]:
#    ax.plot(time_i_grid[dbscan.labels_ == cluster]-shift, window_preds[dbscan.labels_ == cluster]-shift, '.')
#ax.plot(time_i_grid, window_negs, '.', color='black')
#ax.plot(time_i_grid[plateus], window_preds[plateus], '.', color='red')
ax.set_xlabel('Starting time [s]')
ax.set_ylabel('Predicted arrival [s]')
ax.yaxis.set_minor_locator(mtick.MultipleLocator(10))
ax.grid()

cs_norm = cs.data / np.abs(cs.data).max()
fig, ax = plt.subplots()
ax.plot(times, cs_norm, color='black')
for i, ar in enumerate(arrivals_pos[np.argsort(arrivals_pos_qual)][-relevant_preds:]):
    ax.axvline(ar-shift, color='blue', linestyle='--')
    ax.text(ar-shift, 0.1, np.sort(arrivals_pos_qual)[-relevant_preds:][i], rotation=90, fontsize=16)
ax.axvline(ar-shift, color='blue', linestyle='--', label='positive')
for i, ar in enumerate(arrivals_neg[np.argsort(arrivals_neg_qual)][-relevant_preds:]):
   ax.axvline(ar-shift, color='red', linestyle='--')
   ax.text(ar-shift, 0.1, np.sort(arrivals_neg_qual)[-relevant_preds:][i], rotation=90, fontsize=16)
ax.axvline(ar-shift, color='red', linestyle='--', label='negative')
ax.set_ylim(cs_norm.min(), cs_norm.max())
ax.set_xlim(times.min(), times.max())
ax.set_xlabel('Time [s]')
ax.set_ylabel('Amplitude')
#ax.set_title('5caps_wig/0.087_3.96')
ax.xaxis.set_major_locator(mtick.MultipleLocator(500))
ax.xaxis.set_minor_locator(mtick.MultipleLocator(100))
ax.legend(loc='upper right')
fig.tight_layout()
'''
plt.rcParams['animation.ffmpeg_path'] ='/usr/bin/ffmpeg'
import matplotlib.animation as animation
fig, ax = plt.subplots()#figsize=(15,5))
ax.set_ylim(-1, 1)
ax.set_xlim(times.min(), times.max())
ax.set_xlabel('Time (s)')
ax.set_ylabel('Amplitude')
ax.xaxis.set_major_locator(mtick.MultipleLocator(50))
ax.xaxis.set_minor_locator(mtick.MultipleLocator(25))
ax.yaxis.set_major_locator(mtick.MultipleLocator(0.5))
ax.yaxis.set_minor_locator(mtick.MultipleLocator(0.25))
ax.plot(times, cs_norm, color='black')[0]
win_init = ax.axvline(time_i_grid[i]-shift, color='gray')
win_end = ax.axvline(time_f_grid[i]-shift, color='gray')
pred = ax.axvline(window_preds[0]-shift, color='red', linestyle='--')
fig.tight_layout(pad=0.5)
def animate(i):
    i = i
    win_init.set_xdata(time_i_grid[i]-shift)
    win_end.set_xdata(time_f_grid[i]-shift)
    pred.set_xdata(window_preds[i]-shift)
anim = animation.FuncAnimation(fig, animate, interval=1, frames=window_preds.shape[0], repeat=True)
anim.save('sliding_window_crosssec.mp4', writer=animation.FFMpegWriter(fps=120))
'''
'''
fig, ax = plt.subplots(nrows=2, sharex=True)
#axes = [plt.subplot2grid((rows2,cols2), (0,0), colspan=rows2, rowspan=rows2//2, fig=fig2),
#        plt.subplot2grid((rows2,cols2), (rows2//2,0), colspan=rows2, rowspan=rows2//2, fig=fig2)]
cs_norm = cs.data / np.abs(cs.data).max()
#fig, ax = plt.subplots()
ax[0].plot(times, cs_norm, color='black')
for i, ar in enumerate(arrivals_pos[np.argsort(counts_pos)][-2:]):
    ax[0].axvline(ar-shift, color='blue', linestyle='--')
    ax[0].text(ar-5-shift, 0.2, np.sort(counts_pos)[-2:][i], rotation=90, fontsize=16)
ax[0].axvline(ar-shift, color='blue', linestyle='--', label='model')
#for i, ar in enumerate(arrivals_neg[np.argsort(counts_neg)][-5:]):
#   ax.axvline(ar, color='red', linestyle='--')
#   ax.text(ar-5, 0.1, np.sort(counts_neg)[-5:][i], rotation=90, fontsize=16)
#ax.axvline(ar, color='red', linestyle='--', label='negative model')
ax[0].set_ylim(-1, 1)
ax[0].set_xlim(times.min(), times.max())
ax[1].set_xlabel('Time [s]')
ax[0].xaxis.set_major_locator(plt.NullLocator())
ax[0].set_ylabel('Amplitude')
#ax.set_title('5caps_wig/0.087_3.96')
#ax[0].xaxis.set_minor_locator(mtick.MultipleLocator(10))
#ax[0].legend(loc='upper right')

#fighist, axhist = plt.subplots()
ax[1].hist(window_preds-shift, np.arange(begin_time, end_time+0.1, 0.1)-shift, color='black')
for i, cluster in enumerate(clusters[np.argsort(counts_pos)][-2:]):
    ax[1].hist((window_preds-shift)[dbscan.labels_ == cluster], np.arange(begin_time, end_time+0.1, 0.1)-shift, color='red')
ax[1].set_ylabel('Prediction Frequency')
ax[1].xaxis.set_major_locator(mtick.MultipleLocator(50))
ax[1].xaxis.set_minor_locator(mtick.MultipleLocator(25))
fig.tight_layout(pad=1)
fig.subplots_adjust(wspace=0, hspace=0.1)
fig.savefig('../figs/cross_sec_pred.png', dpi=250)
fig.savefig('../figs/cross_sec_pred.svg', dpi=250)
'''