#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 12:11:34 2019

@author: jorgeagr
"""

import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from obspy import read as obspyread

def rescale(data, scaling=(0,1)):
    scaler = MinMaxScaler(feature_range=scaling)
    for i in range(len(data)):
        data[i] = scaler.fit_transform(data[i].reshape(-1,1))
    return data

def decompress(datadir, npydir, model, compression_rate):
    try:
        denoise_dir = 'data/decompressed/' + datadir.split('/')[-2] + '_' + str(compression_rate) + 'x/'
        os.mkdir(denoise_dir)
    except:
        pass
    data = np.load(npydir)
    files = np.sort(os.listdir(datadir))
    for i in range(len(data)):
        decompressed = model.predict(data[i].reshape(1, data.shape[1], 1)).flatten()
        seis = obspyread(datadir + files[i])
        seis[0].data = decompressed
        seis[0].stats.sac.npts = len(decompressed)
        seis[0].stats.sac.delta = 0.1
        seis[0].stats.sac.b += 0.1
        seis.write(denoise_dir+files[i].rstrip('.s_fil') + '.sac')
    
autoencoder = load_model('models/cae_4xcompression_mae_transfer_linear.h5')

datadirs = ['../../seismograms/SS_kept_test/',]
npydirs = ['data/test/test_seismos.npy', ]

for d, n in zip(datadirs, npydirs):
    decompress(d, n, autoencoder, 4)