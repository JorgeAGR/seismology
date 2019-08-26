#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 22:35:00 2018

@author: jorgeagr
"""

import os
import obspy
import numpy as np

# ../../seismology/seismos/SS_kept/

def make_arrays(datadir, th_arrival_var):
    
    name = datadir.split('/')[-2]
    files = os.listdir(datadir)
    
    resample_Hz = 10
    
    seismograms = []
    cut_time = []
    for i, file in enumerate(files):
        try:
            print('File', i+1, '/', len(files), '...')
            seismogram = obspy.read(datadir + file)
            seismogram = seismogram[0].resample(resample_Hz).detrend()
            b = seismogram.stats.sac['b']
            shift = -b
            b = b + shift
            e = seismogram.stats.sac['e'] + shift
            th_arrival = seismogram.stats.sac[th_arrival_var] + shift
            
            if b < th_arrival < e:
                amp = seismogram.data
                time = seismogram.times()
                window_before = 10 # seconds before th_arrival
                window_after = 30 # ditto after ditto
                init = np.where(th_arrival - window_before < time)[0][0]
                end = np.where(th_arrival + window_after > time)[0][-1]
                window_size = (window_before + window_after ) / seismogram.stats.delta
                
                while end-init > window_size:
                    init += 1
                
                while end-init < window_size:
                    end += 1
                
                time_i = time[init]
                time_f = time[end]
                
                if (time_i < th_arrival < time_f):
                    amp_i = amp[init:end]
                    amp_i = (amp_i - amp_i.min()) / (amp_i.max() - amp_i.min())
                    seismograms.append(amp_i)
                    cut_time.append(time_i)
                else:
                    continue
        except:
            continue
        
    seismograms = np.array(seismograms)
    seismograms = seismograms.reshape(len(seismograms), len(seismograms[0]), 1)
    cut_time = np.array(cut_time)
    
    np.save('./pred_data/seismograms_' + name, seismograms)
    np.save('./pred_data/cut_times_' + name, cut_time)
    np.save('./results/file_names_' + datadir.split('/')[-2], np.asarray(files))