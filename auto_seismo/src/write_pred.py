# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 23:31:49 2019

@author: jorge
"""
import os
import obspy
import numpy as np
import scipy as sp
from aux_funcs import check_string

def write_pred(datadir, files, preds):
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
        #seismogram[0].stats.sac.e += shift
        seismogram.write(datadir + file + '.s_fil', format='SAC')
    
    seis = obspy.read(datadir + file + '.s_fil')
    print(seis[0].stats)
    return