#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 14:40:30 2019

@author: jorgeagr
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, UpSampling1D, BatchNormalization, Reshape, Flatten
from keras.models import Model, load_model
from keras.optimizers import Adam
import matplotlib as mpl
from obspy import read as obspyread

# Settings for plots
golden_ratio = (np.sqrt(5) + 1) / 2
width = 12
height = width / golden_ratio

mpl.rcParams['figure.figsize'] = (width, height)
mpl.rcParams['font.size'] = 14
mpl.rcParams['figure.titlesize'] = 'large'
mpl.rcParams['legend.fontsize'] = 'small'
mpl.rcParams['xtick.major.size'] = 12
mpl.rcParams['xtick.minor.size'] = 8
mpl.rcParams['xtick.labelsize'] = 14
mpl.rcParams['ytick.major.size'] = 12
mpl.rcParams['ytick.minor.size'] = 8
mpl.rcParams['ytick.labelsize'] = 14

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
    
autoencoder = load_model('models/rossnet_convautoencoder_denoiser_2sigma_mae_linear.h5')

datadirs = ['../../seismograms/SS_kept/', '../../seismograms/SS_kept_test/']
npydirs = ['data/train/train_seismos_noise_2sigma.npy', 'data/test/test_seismos_noise_2sigma.npy']

for d, n in zip(datadirs, npydirs):
    denoise(d, n, autoencoder)