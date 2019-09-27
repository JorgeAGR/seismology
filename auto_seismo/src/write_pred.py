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

def write_Pred(datadir, files, preds, flipped, pred_var):
    
    for i, file in enumerate(files):
        seismogram = obspy.read(datadir + file + '.s_fil')
        #pred_time = preds[i]
        seismogram[0].stats.sac[pred_var] = preds[i]
        if flipped[i]:
            None
            #seismogram[0].data = -seismogram[0].data
        
        seismogram[0].stats.sac[pred_var] = shift_Max(seismogram[0], pred_var)
        seismogram.write(datadir + file + '.s_fil', format='SAC')
    
    return

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