#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 20:29:44 2019

@author: jorgeagr
"""
import os
import obspy
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.animation as animation
from tensorflow.keras.models import load_model

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

directory = '../../../seismograms/SS_kept/'
dir_name = directory.split('/')[-2] + '/'
try:
    os.makedirs('../figs/etc/' + dir_name)
except:
    pass

seis_files = np.sort(os.listdir(directory))
model = 'SS2040'
model = load_model('../../seyesmolite/models/{}/{}.h5'.format(model, model))
#model = load_model('../pickerlite/models/SS_40_model.h5')
wb = 20
wa = 40
tot_time = (wb+wa)*10
rand = 0

file = os.listdir('/home/jorgeagr/Documents/seismograms/SS_kept/')[50]
seismogram = obspy.read(directory+file)[0]
time = seismogram.times()
init = np.where(time > (seismogram.stats.sac.t2 - 100 - seismogram.stats.sac.b))[0][0]
end = len(time)#np.where(time > (seismogram.stats.sac.t2 + 100 - seismogram.stats.sac.b))[0][0]
shift = time[init]
time = time[init:end] + seismogram.stats.sac.b# - time[init]
amp_i = seismogram.data[init:end]
#amp_i = (amp_i - amp_i.min()) / (amp_i.max() - amp_i.min())
amp_i = amp_i / np.abs(amp_i).max()
# for test data
#pred = seismogram.stats.sac.t7 + seismogram.stats.sac.b
actual = seismogram.stats.sac.t6# - seismogram.stats.sac.b
# if using unknown data
theoretical = seismogram.stats.sac.t2 + rand# - seismogram.stats.sac.b
data = seismogram.data[int(theoretical - seismogram.stats.sac.b - wb)*10:int(theoretical - seismogram.stats.sac.b + wa)*10]
pred = model.predict(data.reshape(1,tot_time,1)/np.abs(data.max()))[0].flatten() + int(theoretical) - wb
fig, ax = plt.subplots()
ax.plot(time, amp_i, color='black')
ax.axvline(theoretical, color='gray', linestyle='--', label='Theory')
#if actual:
ax.axvline(pred, color='blue', linewidth=1.5, label='Prediction')
ax.axvline(actual, color='red', linewidth=0.8, linestyle='--', label='Actual')
ax.set_xlim(time[0], time[-1])
ax.set_ylim(-1.05, 1.05)
ax.xaxis.set_major_locator(mtick.MultipleLocator(50))
ax.yaxis.set_major_locator(mtick.MultipleLocator(0.5))
ax.xaxis.set_minor_locator(mtick.MultipleLocator(25))
ax.yaxis.set_minor_locator(mtick.MultipleLocator(0.25))
ax.set_xlabel('Time [s]')
ax.set_ylabel('Relative Amplitude')
ax.legend()
plt.tight_layout()
#plt.close()
#fig.savefig('../figs/etc/' + dir_name + 'pred_' + file + '.png', dpi=500)
#fig.savefig('../figs/etc/' + dir_name + 'pred_' + file + '.svg', dpi=500)
