# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 00:17:45 2019

@author: jorge
"""

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
import pandas as pd
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

cap = '5'
file = 'n35_128.10'

pred_file = pd.read_csv('cross_secs_dat/model_pred/'+cap+'caps_deg_results.csv')

# By indexing with [similar_inds][similar_sort], this returns a subset of
# quality factors sorted from ratios 0.9 to 1.2, index 212 is the first ratio=1
# Try to take a sample in increments of 0.1 with as close a ratio to 1
qual_similarity = pred_file['410qual'].values / pred_file['660qual'].values
similar_inds = np.where((qual_similarity > 0.9) & (qual_similarity < 1.1))
similar_sort = np.argsort(qual_similarity[similar_inds])
#quality_sort = np.argsort(pred_file['410qual'].values[similar_inds][qs_sort])

qual_dic = {'0.1': 173, '0.2': 219, '0.3': 81,'0.4': 233, '0.5': 264,
            '0.6': 203,'0.7': 224,'0.8': 191,'0.9': 209,'1.0': 296}






'''
cs = obspy.read('../../seismograms/cross_secs/' + cap + 'caps_deg/' + file + '.sac')

with open('cross_secs_dat/model_pred/'+cap+'caps_deg_results.csv') as pred_csv:
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
qual_410 = float(pred_line.split(',')[4])
qual_660 = float(pred_line.split(',')[8])
quals = [qual_660, qual_410]
try:
    lauren_410 = get_Lauren_Pred(cap, '410', file)# - 4.2 # temp 4.2 shift from error by lauren
    lauren_660 = get_Lauren_Pred(cap, '660', file)#ls - 4.2
except:
    pass

cs_norm = cs.data / np.abs(cs.data).max()
fig, ax = plt.subplots()
ax.plot(times, cs_norm, color='black')
for i, ar in enumerate([arr_660, arr_410]):
    print(ar)
    ax.axvline(ar, color='blue', linestyle='--')
    ax.text(ar-15, 0.1, quals[i], rotation=90, fontsize=16)
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
ax.axvline(ar, color='red', linestyle='--', label='negative model')
ax.set_ylim(cs_norm.min(), cs_norm.max())
ax.set_xlim(times.min(), times.max())
ax.set_xlabel('Time [s]')
ax.set_ylabel('Amplitude')
ax.set_title(cap + 'cap/' + file)
ax.xaxis.set_minor_locator(mtick.MultipleLocator(10))
#ax.legend(loc='upper right')
fig.tight_layout()
plt.show()
'''