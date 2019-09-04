# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 11:30:36 2019

@author: jorge
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 22:35:00 2018

@author: jorgeagr
"""

import os
import obspy
import numpy as np
from aux_funcs import check_string

datadir = '../train_data/'
good = 'good/' # qual_var 1
bad = 'bad/' # qual_var 0

def make_arrays(datadir, qualtype, th_arrival_var, arrival_var, qual_var, wave_type):
    
    resample_Hz = 10
    
    files = np.sort(os.listdir(datadir))
    seismograms = []
    arrivals = []
    cut_time = []
    file_names = []
    #polarities = []
    
    for i, file in enumerate(files):
        file = check_string(file)
        print(i+1, '/', len(files))
        seismogram = obspy.read(datadir + file)
        seismogram = seismogram[0].resample(resample_Hz).detrend()
        b = seismogram.stats.sac['b']
        shift = -b
        b = b + shift
        e = seismogram.stats.sac['e'] + shift
        th_arrival = seismogram.stats.sac[th_arrival_var] + shift
        arrival = seismogram.stats.sac[arrival_var] + shift
        
        if not (b < th_arrival < e):
            th_arrival = arrival - 2
        
        if b < arrival < e:
            amp = seismogram.data
            time = seismogram.times()
            window_before = 10 # seconds before th_arrival
            window_after = 30 # ditto after ditto
            rand_window = np.random.rand(6)
            rand_window[0] = 0
            for j, n in enumerate(rand_window):
                '''
                pol = 1
                if j > 2:
                    amp = -amp
                    pol = 0
                '''
                rand_arrival = th_arrival - n*2
                init = np.where(rand_arrival - window_before < time)[0][0]
                end = np.where(rand_arrival + window_after > time)[0][-1]
                window_size = (window_before + window_after ) / seismogram.stats.delta
                
                while end-init > window_size:
                    end += -1
                
                while end-init < window_size:
                    end += 1
                
                time_i = time[init]
                time_f = time[end]
                if (time_i < arrival < time_f) and (time_i < rand_arrival < time_f):
                    amp_i = amp[init:end]
                    amp_i = (amp_i - amp_i.min()) / (amp_i.max() - amp_i.min())
                    seismograms.append(amp_i)
                    arrivals.append(arrival - time[init])
                    cut_time.append(time[init])
                    file_names.append(file)
                    #polarities.append(pol)
                else:
                    continue
        
    
    seismograms = np.array(seismograms).reshape(len(seismograms), len(seismograms[0]), 1)
    arrivals = np.array(arrivals).reshape(len(arrivals), 1)
    cut_time = np.array(cut_time)
    qualities = np.ones(len(files)) * qual_var
    qualities = qualities.reshape(len(qualities), 1)
    file_names = np.array(file_names)
    #polarities = np.array(polarities)
    
    np.save('../train_data/seismograms_' + wave_type, seismograms)
    np.save('../train_data/arrivals_' + wave_type, arrivals)
    np.save('../train_data/cut_times_' + wave_type, cut_time)
    #np.save('../train_data/quality_' + wave_type, qualities)
    np.save('../train_data/file_names_' + wave_type, file_names)
    #np.save('../train_data/polarities_' + wave_type, polarities)

datadir = '../../../seismograms/SS_kept/'
qualtype = 'good'
th_arrival_var = 't2'
arrival_var = 't6'
qual_var = 1
wave_type = 'SS'
make_arrays(datadir, qualtype, th_arrival_var, arrival_var, qual_var, wave_type)
#make_arrays(bad, 't2')
'''
Deprecated version, just in case

import os
import obspy
import numpy as np
#from scipy.interpolate import interp1d

datadir = '../train_data/'
good = 'good/' # qual_var 1
bad = 'bad/' # qual_var 0

def make_arrays(datadir, qualtype, th_arrival_var, arrival_var, qual_var):
    
    resample_Hz = 10
    
    files = os.listdir(datadir)
    seismograms = []
    arrivals = []
    cut_time = []
    
    for i, file in enumerate(files):
        print(i, '/', len(files))
        seismogram = obspy.read(datadir + file)
        seismogram = seismogram[0].resample(resample_Hz).detrend()
        #N = seismogram[0].stats.npts
        #delta = np.round(seismogram.stats.delta, 1)
        b = seismogram.stats.sac['b']
        shift = -b
        b = b + shift
        e = seismogram.stats.sac['e'] + shift
        th_arrival = seismogram.stats.sac[th_arrival_var] + shift
        arrival = seismogram.stats.sac[arrival_var] + shift
        
        if not (b < th_arrival < e):
            th_arrival = arrival - 2
        
        if b < arrival < e:
            amp = seismogram.data
            time = seismogram.times()
            if len(time) > len(amp):
                time = time[:len(amp)]
            window_time = 20 # seconds after th_arrival
            rand_window = np.random.rand(5)
            rand_window[np.argmax(rand_window)] = 0
            for n in rand_window:
                rand_arrival = th_arrival - n*2
                init = np.where(rand_arrival < time)[0][0]
                end = np.where(rand_arrival + window_time > time)[0][-1]
                amp_i = amp[init:end]
                window_size = window_time / seismogram.stats.delta
                
                while amp_i.size > window_size:
                    init += 1
                    amp_i = amp[init:end]
                
                while amp_i.size < window_size:
                    end += 1
                    amp_i = amp[init:end]
                
                amp_i = (amp_i - amp_i.min()) / (amp_i.max() - amp_i.min())
                seismograms.append(amp_i)
                arrivals.append(arrival - rand_arrival)
                cut_time.append(rand_arrival)
        
            
    seismograms = np.array(seismograms)
    seismograms = seismograms.reshape(len(seismograms), len(seismograms[0]), 1)
    arrivals = np.array(arrivals)
    arrivals = arrivals.reshape(len(arrivals), 1)
    cut_time = np.array(cut_time)
    qualities = np.ones(len(files)) * qual_var
    qualities = qualities.reshape(len(qualities), 1)
    
    np.save('./train_data/seismograms_' + th_arrival_var, seismograms)
    np.save('./train_data/arrivals_' + th_arrival_var, arrivals)
    np.save('./train_data/cut_times_' + th_arrival_var, cut_time)
    np.save('./train_data/quality_' + th_arrival_var, qualities)

datadir = '../../seismology/seismos/SS_kept/'
qualtype = 'good'
th_arrival_var = 't2'
arrival_var = 't6'
qual_var = 1
make_arrays(datadir, qualtype, th_arrival_var, arrival_var, qual_var)
#make_arrays(bad, 't2')
'''