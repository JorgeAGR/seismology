# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 14:23:29 2019

@author: jorge
"""

import os
import obspy
import numpy as np
from aux_funcs import check_String

def training_Arrays(datadir, qualtype, th_arrival_var, arrival_var, 
                    qual_var, wave_type, window_before=10, window_after=30):
    
    resample_Hz = 10
    
    files = np.sort(os.listdir(datadir))
    seismograms = []
    seismograms_flipped = []
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
            th_arrival = arrival - 15 * np.random.rand()
        
        if b < arrival < e:
            amp = seismogram.data
            time = seismogram.times()
            rand_window_shifts = 2*np.random.rand(6) - 1 # [-1, 1] shift multiplier
            abs_sort = np.argsort(np.abs(rand_window_shifts))
            rand_window_shifts = rand_window_shifts[abs_sort]
            rand_window_shifts[0] = 0
            for j, n in enumerate(rand_window_shifts):
                rand_arrival = th_arrival - n*5
                init = np.where(np.round(rand_arrival - window_before, 1) == time)[0][0]
                end = np.where(np.round(rand_arrival + window_after, 1) == time)[0][0]
                '''
                while end-init > window_size:
                    end += -1
                
                while end-init < window_size:
                    end += 1
                '''
                time_i = time[init]
                time_f = time[end]
                '''
                while rand_arrival < time[init]:
                    rand_arrival += 0.1
                '''
                if not (time_i < arrival < time_f):
                    time_i = arrival - 15 * np.random.rand() - window_before
                    time_f = time_i + window_after
                
                amp_p = amp[init:end]
                amp_n = -amp_p
                # Rescale to [0, 1]
                #amp_p = (amp_p - amp_p.min()) / (amp_p.max() - amp_p.min())
                #amp_n = (amp_n - amp_n.min()) / (amp_n.max() - amp_n.min())
                # Normalize by absolute peak, [-1, 1]
                amp_p = amp_p / np.abs(amp_p).max()
                amp_n = amp_n / np.abs(amp_n).max()
                seismograms.append(amp_p)
                seismograms_flipped.append(amp_n)
                arrivals.append(arrival - time[init])
                cut_time.append(time[init])
                file_names.append(file)
        
    
    seismograms = np.array(seismograms).reshape(len(seismograms), len(seismograms[0]), 1)
    seismograms_flipped = np.array(seismograms_flipped).reshape(len(seismograms_flipped), 
                                   len(seismograms_flipped[0]), 1)
    arrivals = np.array(arrivals).reshape(len(arrivals), 1)
    cut_time = np.array(cut_time)
    qualities = np.ones(len(files)) * qual_var
    qualities = qualities.reshape(len(qualities), 1)
    file_names = np.array(file_names)
    
    np.save('./train_data/seismograms_' + wave_type, seismograms)
    np.save('./train_data/seismograms_flipped_' + wave_type, seismograms_flipped)
    np.save('./train_data/arrivals_' + wave_type, arrivals)
    np.save('./train_data/cut_times_' + wave_type, cut_time)
    np.save('./train_data/file_names_' + wave_type, file_names)

def make_Arrays(datadir, th_arrival_var, window_before=10, window_after=30):
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
            rand_window_shifts = 2*np.random.rand(10) - 1 # [-1, 1] shift multiplier
            abs_sort = np.argsort(np.abs(rand_window_shifts))
            rand_window_shifts = rand_window_shifts[abs_sort]
            rand_window_shifts[0] = 0
            rand_window_shifts[5] = 0
            for j, n in enumerate(rand_window_shifts):
                if j > 4:
                    amp = -amp
                rand_arrival = th_arrival - n*5
                init = np.where(np.round(rand_arrival - window_before, 1) == time)[0][0]
                end = np.where(np.round(rand_arrival + window_after, 1) == time)[0][0]
                
                time_i = time[init]
                
                amp_i = amp[init:end]
                # Rescale to [0, 1]
                #amp_i = (amp_i - amp_i.min()) / (amp_i.max() - amp_i.min())
                # Normalize by absolute peak
                amp_i = amp_i / np.abs(amp_i).max()
                cut_times.append(time_i)
                theoreticals.append(th_arrival - time_i)
                if j > 4:
                    flips.append(amp_i)
                else:
                    noflips.append(amp_i)
                    
            np.savez('./pred_data/' + name + '/' + file.rstrip('.s_fil'), 
                 noflips=noflips, flips=flips, cuts=cut_times, theory=theoreticals)