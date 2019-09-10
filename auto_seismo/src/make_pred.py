#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 14:22:23 2019

@author: jorgeagr
"""
import os
import numpy as np
from aux_funcs import check_String

def predict_Arrival(model, datadir, debug_mode=False):
    name = datadir.split('/')[-2]
    name = check_String(name)
    npzdir = './pred_data/' + name + '/'
    
    files = np.sort(os.listdir(npzdir))
    if debug_mode:
        files = files[:100]
    
    seis_names = []
    pred_arrival = []
    pred_error = []
    flipped = []
    for file in files:
        file = check_String(file)
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

def predict_Train_Data(model, datadir, debug_mode=False):
    
    name = datadir.split('/')[-2]
    name = check_String(name)
    npzdir = './pred_data/' + name + '/'
    
    files = np.sort(os.listdir(npzdir))
    if debug_mode:
        files = files[:100]
    
    seis_names = []
    pred_arrival = []
    for file in files:
        file = check_String(file)
        seismogram = np.load(npzdir + file)
        
        noflip = seismogram['noflips'][0]
        noflip_cut_time = seismogram['cuts'][0]
        
        #pred = arrive_model.predict(np.load(file)).flatten()
        noflip_pred = model.predict(np.reshape(noflip, 
                                                (1, len(noflip), 1))).flatten()
        
        seis_names.append(file.rstrip('.npz'))
        pred_arrival.append(noflip_pred + noflip_cut_time)
        
    seis_names = np.asarray(seis_names)
    pred_arrival = np.asarray(pred_arrival)
    
    return seis_names, pred_arrival