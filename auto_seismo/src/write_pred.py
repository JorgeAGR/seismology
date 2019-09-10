# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 23:31:49 2019

@author: jorge
"""
import os
import obspy
import numpy as np
import scipy as sp
from aux_funcs import check_String

def write_Pred(datadir, files, preds, flipped):
    #name = datadir.split('/')[-2]
    #name = check_string(name)
    #npzdir = './pred_data/' + name + '/'
    
    #files = np.sort(os.listdir(npzdir))
    for i, file in enumerate(files):
        seismogram = obspy.read(datadir + file + '.s_fil')
        #maxima_times = seismogram[0].times()[sp.signal.argrelmax(seismogram[0].data)]
        pred_time = preds[i]
        #pred_time = maxima_times[np.argmin(np.abs(maxima_times - pred_time))]
        seismogram[0].stats.sac.t6 = pred_time
        if flipped[i]:
            None
            #seismogram[0].data = -seismogram[0].data
        #seismogram[0].stats.sac.e += shift
        
        seismogram[0].stats.sac.t6 = shift_Max(seismogram[0])
        
        
        seismogram.write(datadir + file + '.s_fil', format='SAC')
    
    return

def shift_Max(seis):
    data = seis.data
    time = seis.times()
    arrival = 0
    new_arrival = seis.stats.sac.t6
    #for i in range(3):
    while (new_arrival - arrival) != 0:
        arrival = new_arrival
        init = np.where(time > (arrival - 1 - seis.stats.sac.b))[0][0]
        end = np.where(time > (arrival + 1 - seis.stats.sac.b))[0][0]
        
        amp_max = np.argmax(np.abs(data[init:end]))
        time_ind = np.arange(init, end, 1)[amp_max]
        
        new_arrival = time[time_ind]
        #print(new_arr)
        #if np.abs(new_arr - arrival) < 1e-2:
        #    break
    return arrival