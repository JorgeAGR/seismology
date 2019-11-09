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
    

cap = '15'
file = '0.261_0'

cs = obspy.read('../../seismograms/cross_secs/' + cap + 'caps_wig/' + file + '.sac')

def get_Lauren_Pred(cap, precursor, csbin):
    file_path = 'cross_secs_dat/lauren_pred/' + cap + 'caps_S' + precursor + 'S.dat'
    with open(file_path) as datfile:
        dat_bins = np.asarray([line.split(' ')[0] for line in datfile])
    dat_times = np.loadtxt(file_path, usecols=(1, 2))
    ind = np.where(dat_bins == csbin)[0][0]
    return dat_times[ind,0]#, dat_times[ind,1]

def get_Lauren_Pred_Bootstraps(cap, precursor, csbin):
    '''
    As per Lauren:
    "Columns are: Time, slowness, mean amplitude value, standard deviation
    So you will use $4 as the error measurement. $3 should correspond to the original cross-section, but may not for messy data."
    '''
    if precursor == '410':
        min_t, max_t = -170, -140
    elif precursor == '660':
        min_t, max_t = -240, -210
    
    cap_path = 'cross_secs_dat/other_lauren/' + cap + 'caps/'
    bin_file = csbin
    bin_file += '_bootstrap.dat'
    dat_file = np.loadtxt(cap_path + bin_file)
    ind_range = np.asarray([i for i in range(len(dat_file[:,0])) if min_t < dat_file[i,0] < max_t])
    arrival_ind = dat_file[ind_range,2].argmax()
    dat_time = dat_file[ind_range,0][arrival_ind] # arrival time
    dat_error = dat_file[ind_range,3][arrival_ind] # error?? of what?
    
    return dat_time#, dat_error

with open('cross_secs_dat/model_pred/'+cap+'caps_wig_preds.csv') as pred_csv:
    for line in pred_csv:
        if file == line.split(',')[0].rstrip('.sac'):
            pred_line = line
            break

cs = cs[0].resample(10)

shift = -cs.stats.sac.b
times = cs.times() - shift
b = cs.stats.sac.b + shift
e = cs.stats.sac.e + shift
arr_410 = float(pred_line.split(',')[1])
arr_660 = float(pred_line.split(',')[5])
try:
    lauren_410 = get_Lauren_Pred_Bootstraps(cap, '410', file)# - 4.2 # temp 4.2 shift from error by lauren
    lauren_660 = get_Lauren_Pred_Bootstraps(cap, '660', file)#ls - 4.2
except:
    pass

cs_norm = cs.data / np.abs(cs.data).max()
fig, ax = plt.subplots()
ax.plot(times, cs_norm, color='black')
for i, ar in enumerate([arr_660, arr_410]):
    print(ar)
    ax.axvline(ar, color='blue', linestyle='--')
    #ax.text(ar-5, 0.1, np.sort(counts_pos)[-2:][i], rotation=90, fontsize=16)
ax.axvline(ar, color='blue', linestyle='--', label='model')
try:
    for i, ar in enumerate([lauren_660, lauren_410]):
        ax.axvline(ar, color='red', linestyle=':')
    ax.axvline(ar, color='red', linestyle=':', label='lauren')
except:
    pass
#for i, ar in enumerate(arrivals_neg[np.argsort(counts_neg)][-5:]):
#   ax.axvline(ar, color='red', linestyle='--')
#   ax.text(ar-5, 0.1, np.sort(counts_neg)[-5:][i], rotation=90, fontsize=16)
#ax.axvline(ar, color='red', linestyle='--', label='negative model')
ax.set_ylim(cs_norm.min(), cs_norm.max())
ax.set_xlim(times.min(), times.max())
ax.set_xlabel('Time [s]')
ax.set_ylabel('Amplitude')
ax.set_title(cap + 'cap/' + file)
ax.xaxis.set_minor_locator(mtick.MultipleLocator(10))
#ax.legend(loc='upper right')
fig.tight_layout()
plt.show()