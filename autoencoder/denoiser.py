#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 14:40:30 2019

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

def denoise(datadir, npydir, model):
    try:
        denoise_dir = 'data/denoised/' + datadir.split('/')[-2] + '/'
        os.mkdir(denoise_dir)
    except:
        pass
    data = np.load(npydir)
    files = np.sort(os.listdir(datadir))
    for i in range(len(data)):
        denoised = model.predict(data[i].reshape(1, data.shape[1], 1)).flatten()
        seis = obspyread(datadir + files[i])
        seis[0].data = denoised
        seis[0].stats.sac.npts = len(denoised)
        seis[0].stats.sac.delta = 0.1
        seis[0].stats.sac.b += 0.1
        seis.write(denoise_dir+files[i].rstrip('.s_fil') + '.sac')
    
autoencoder = load_model('models/cae_denoiser_2sigma_mae_transfer_linear.h5')

datadirs = ['../../seismograms/SS_kept_test/',]
npydirs = ['data/test/test_seismos_noise_2sigma.npy', ]

for d, n in zip(datadirs, npydirs):
    denoise(d, n, autoencoder)