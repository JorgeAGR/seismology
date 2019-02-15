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

datadir = '../train_data/'
good = 'good/' # qual_var 1
bad = 'bad/' # qual_var 0

def make_arrays(datadir, qualtype, arrival_var, qual_var):
    
    files = os.listdir(datadir)
    seismograms = []
    arrivals = []
    cut_time = []
    
    for i, file in enumerate(files):
        print(i, '/', len(files))
        seismogram = obspy.read(datadir + file)
        #N = seismogram[0].stats.npts
        delta = np.round(seismogram[0].stats.delta, 1)
        b = seismogram[0].stats.sac['b']
        shift = -b
        b = b + shift
        e = seismogram[0].stats.sac['e'] + shift
        arrival = seismogram[0].stats.sac[arrival_var] + shift
        
        if b < arrival < e:
            amp = seismogram[0].data
            #amp = (amp - np.min(amp)) / (np.max(amp) - np.min(amp))
            time = np.arange(b, e + delta, delta)
            if len(time) > len(amp):
                time = time[:len(amp)]
            rand_window = np.random.rand(5)
            for n in rand_window:
                init = np.where(arrival-10-10*n < time)[0][0]
                end = np.where(arrival+30-30*n > time)[0][-1]
                amp_i = amp[init:end]
                '''
                while True:
                    if end-init < 400:
                        end += 1
                    elif end-init > 400:
                        init += 1
                    else:
                        break
                '''
                while amp_i.size > 400:
                    init += 1
                    amp_i = amp[init:end]
                
                while amp_i.size < 400:
                    end += 1
                    amp_i = amp[init:end]
                
                #amp_i = amp_i/np.abs(np.mean(amp_i))
                amp_i = (amp_i - np.min(amp_i)) / (np.max(amp_i) - np.min(amp_i))
                #amp_i = (amp_i - np.mean(amp_i)) / np.std(amp_i)
                seismograms.append(amp_i)
                arrivals.append(arrival - time[init])
                cut_time.append(time[init])
        
        
    seismograms = np.array(seismograms)
    seismograms = seismograms.reshape(len(seismograms), len(seismograms[0]), 1)
    arrivals = np.array(arrivals)
    arrivals = arrivals.reshape(len(arrivals), 1)
    cut_time = np.array(cut_time)
    qualities = np.ones(len(files)) * qual_var
    qualities = qualities.reshape(len(qualities), 1)
    
    np.save('../train_data/seismogramswhole_' + qualtype, seismograms)
    np.save('../train_data/arrivalswhole_' + qualtype, arrivals)
    np.save('../train_data/cut_times_' + qualtype, cut_time)
    np.save('../train_data/quality_' + qualtype, qualities)

datadir = '../../seismos/SS_kept/'
qualtype = 'good'
arrival_var = 't6'
qual_var = 1
make_arrays(datadir, qualtype, arrival_var, qual_var)
#make_arrays(bad, 't2')