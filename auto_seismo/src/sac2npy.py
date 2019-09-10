# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 14:23:29 2019

@author: jorge
"""

import os
import obspy
import numpy as np
from aux_funcs import check_String

def training_Arrays(datadir, qualtype, th_arrival_var, arrival_var, qual_var, wave_type):
    
    resample_Hz = 10
    
    files = np.sort(os.listdir(datadir))
    seismograms = []
    arrivals = []
    cut_time = []
    file_names = []
    
    for i, file in enumerate(files):
        file = check_String(file)
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
                else:
                    continue
        
    
    seismograms = np.array(seismograms).reshape(len(seismograms), len(seismograms[0]), 1)
    arrivals = np.array(arrivals).reshape(len(arrivals), 1)
    cut_time = np.array(cut_time)
    qualities = np.ones(len(files)) * qual_var
    qualities = qualities.reshape(len(qualities), 1)
    file_names = np.array(file_names)
    
    np.save('./train_data/seismograms_' + wave_type, seismograms)
    np.save('./train_data/arrivals_' + wave_type, arrivals)
    np.save('./train_data/cut_times_' + wave_type, cut_time)
    np.save('./train_data/file_names_' + wave_type, file_names)

#datadir = '../../../seismograms/SS_kept/'
#qualtype = 'good'
#th_arrival_var = 't2'
#arrival_var = 't6'
#qual_var = 1
#wave_type = 'SS'
#make_arrays(datadir, qualtype, th_arrival_var, arrival_var, qual_var, wave_type)

def make_Arrays(datadir, th_arrival_var):
    name = datadir.split('/')[-2]
    name = check_String(name)
    files = np.sort(os.listdir(datadir))
    if name not in os.listdir('./pred_data/'):
        os.makedirs('./pred_data/' + name)
    
    resample_Hz = 10
    for i, file in enumerate(files):
        file = check_String(file)
        noflips = []
        flips = []
        cut_times = []
        theoreticals = []
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
                rand_window_shifts = np.random.rand(10)
                rand_window_shifts[0] = 0
                for j, n in enumerate(rand_window_shifts):
                    if j > 4:
                        amp = -amp
                    rand_arrival = th_arrival - n*8
                    init = np.where(rand_arrival - window_before < time)[0][0]
                    end = np.where(rand_arrival + window_after > time)[0][-1]
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
                        cut_times.append(time_i)
                        theoreticals.append(th_arrival - time_i)
                        if j > 4:
                            flips.append(amp_i)
                        else:
                            noflips.append(amp_i)
                        #seismograms.append(amp_i)
                        #cut_time.append(time_i)
                    else:
                        continue
                np.savez('./pred_data/' + name + '/' + file.rstrip('.s_fil'), 
                     noflips=noflips, flips=flips, cuts=cut_times, theory=theoreticals)    
        except:
            continue