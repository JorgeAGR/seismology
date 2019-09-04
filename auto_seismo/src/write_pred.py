# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 23:31:49 2019

@author: jorge
"""
import os
import obspy
import numpy as np
from aux_funcs import check_string

def write_pred(datadir, files, preds):
    #name = datadir.split('/')[-2]
    #name = check_string(name)
    #npzdir = './pred_data/' + name + '/'
    
    #files = np.sort(os.listdir(npzdir))
    for i, file in enumerate(files):
        seismogram = obspy.read(datadir + file + '.s_fil')
        shift = -preds[i]
        seismogram[0].stats.sac.b += shift
        seismogram[0].stats.sac.t6 = 0
        seismogram[0].stats.sac.e += shift
        seismogram.write(datadir + file + '.s_fil', format='SAC')
    return