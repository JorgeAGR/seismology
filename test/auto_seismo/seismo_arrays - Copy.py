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


def make_arrays(datadir, arrival_var):
    
    name = datadir.split('/')[-2]
    files = os.listdir(datadir)
    
    seismograms = []
    cut_time = []
    file_list = []
    for i, file in enumerate(files):
        try:
            print('File', i+1, '/', len(files), '...')
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
                time = np.arange(b, e + delta, delta)
                if len(time) > len(amp):
                    time = time[:len(amp)]
                init = np.where(arrival < time)[0][0]
                end = np.where(arrival+40 > time)[0][-1]
                amp_i = amp[init:end]
                
                while amp_i.size > 400:
                    init += 1
                    amp_i = amp[init:end]
                
                while amp_i.size < 400:
                    end += 1
                    amp_i = amp[init:end]
                
                amp_i = (amp_i - np.min(amp_i)) / (np.max(amp_i) - np.min(amp_i))
                #seismograms[i] = amp_i.reshape(len(amp_i), 1)
                seismograms.append(amp_i)
                cut_time.append(time[init])
                file_list.append(file)
        except:
            continue
        
    seismograms = np.array(seismograms)
    seismograms = seismograms.reshape(len(seismograms), len(seismograms[0]), 1)
    cut_time = np.array(cut_time)
    
    np.save('./pred_data/seismograms_' + name, seismograms)
    np.save('./pred_data/cut_times_' + name, cut_time)
    np.save('./results/file_names_' + datadir.split('/')[-2], np.asarray(file_list))