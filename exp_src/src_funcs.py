#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 22:33:11 2019

@author: jorgeagr
"""
import os
import obspy
import numpy as np
import keras.backend as tfb

def check_String(string):
    return string.rstrip('\r')

def abs_Error(y_true, y_pred):
    return tfb.mean(tfb.abs(y_true - y_pred))

def predict_Arrival(model, datadir, debug_mode=False, files=None):
    name = datadir.split('/')[-2]
    name = check_String(name)
    npzdir = '../auto_seismo/pred_data/' + name + '/'
    
    if files is None:
        files = np.sort(os.listdir(npzdir))

    if debug_mode:
        files = files[:100]
    
    seis_names = []
    pred_arrival = []
    pred_error = []
    flipped = []
    for file in files:
        file = check_String(file)
        file = file.rstrip('.s_fil') + '.npz'
        seismogram = np.load(npzdir + file)
        
        noflips = seismogram['noflips']
        flips = seismogram['flips']
        cutoff = len(noflips)
        noflips_cut_times = seismogram['cuts'][:cutoff]
        noflips_th_arrivals = seismogram['theory'][:cutoff]
        flips_cut_times = seismogram['cuts'][cutoff:]
        flips_th_arrivals = seismogram['theory'][cutoff:]
        
        #pred = arrive_model.predict(np.load(file)).flatten()
        noflip_preds = model.predict(np.reshape(noflips, 
                                                (len(noflips), len(noflips[0]), 1))).flatten()
        flip_preds = model.predict(np.reshape(flips, 
                                             (len(flips), len(flips[0]), 1))).flatten()
        
        pred_means = [np.mean(noflip_preds + noflips_cut_times), 
                      np.mean(flip_preds + flips_cut_times)]
        pred_stds = [np.std(noflip_preds + noflips_cut_times), 
                     np.std(flip_preds + flips_cut_times)]
        
        if 0.5 > pred_stds[0]/pred_stds[1] > 1.5:
            correct_pred = np.argmin(pred_means)
        else:
            correct_pred = np.argmin(pred_stds)
        #th_arrival = noflips_th_arrivals[0] + noflips_cut_times[0]
        pred_time = pred_means[correct_pred]
        
        seis_names.append(file.rstrip('.npz'))
        pred_arrival.append(pred_time)
        pred_error.append(pred_stds[correct_pred])
        flipped.append(correct_pred)
        
    seis_names = np.asarray(seis_names)
    pred_arrival = np.asarray(pred_arrival)
    pred_error = np.asarray(pred_error)
    flipped = np.asarray(flipped)
    
    return seis_names, pred_arrival, pred_error, flipped