#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 14:22:23 2019

@author: jorgeagr
"""
import os
import numpy as np
from aux_funcs import check_string

def predict_arrival(model, datadir):
    name = datadir.split('/')[-2]
    name = check_string(name)
    npzdir = './pred_data/' + name + '/'
    
    seis_names = []
    pred_arrival = []
    pred_error = []
    flipped = []
    for file in os.listdir(npzdir):
        file = check_string(file)
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
        
        correct_pred = np.argmin(pred_stds)
        th_arrival = noflips_th_arrivals[0] + noflips_cut_times[0]
        
        seis_names.append(file.rstrip('.npz'))
        pred_arrival.append(pred_means[correct_pred])
        pred_error.append(pred_stds[correct_pred])
        flipped.append(correct_pred)
        
    seis_names = np.asarray(seis_names)
    pred_arrival = np.asarray(pred_arrival)
    pred_error = np.asarray(pred_error)
    flipped = np.asarray(flipped)
    
    return seis_names, pred_arrival, pred_error, flipped

def predict_train_data(model, traindir):
    return    