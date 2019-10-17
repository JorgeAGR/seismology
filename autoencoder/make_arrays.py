#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 23:05:08 2019

@author: jorgeagr
"""

import os
import numpy as np
import obspy

def sac2npy(directory, files):
    seismos = np.zeros((len(files), data_points))
    for i, file in enumerate(files):
        print('File', i+1, '/', len(files))
        seis = obspy.read(directory + file)
        seis = seis[0].resample(resample_Hz).detrend()
        seis.data = seis.data / np.abs(seis.data).max()
        if seis.data.shape[0] > data_points:
            seis.data = seis.data[1:]
        seismos[i] = seis.data
    
    return seismos

directories = ['data/', 'data/train/', 'data/test/']
for d in directories:
    try:
        os.mkdir(d)
    except:
        pass

train_dir = '../../../seismograms/SS_kept/'
test_dir = '../../../seismograms/SS_kept_test/'
train_files = np.sort(os.listdir(train_dir))
test_files = np.sort(os.listdir(test_dir))

resample_Hz = 10
data_points = 5000

train_seismos = sac2npy(train_dir, train_files)
test_seismos = sac2npy(test_dir, test_files)

train_seismos = train_seismos.reshape(train_seismos.shape[0], train_seismos.shape[1], 1)
test_seismos = test_seismos.reshape(test_seismos.shape[0], test_seismos.shape[1], 1)

np.save('data/train/train_seismos', train_seismos)
np.save('data/test/test_seismos', test_seismos)