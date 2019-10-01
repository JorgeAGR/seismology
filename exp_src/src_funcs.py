#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 14:22:23 2019

@author: jorgeagr
"""
import os
import numpy as np
from aux_funcs import check_String


def make_Pred(model, seismogram_npz):
    noflips = seismogram_npz['noflips']
    flips = seismogram_npz['flips']
    cutoff = len(noflips)
    noflips_cut_times = seismogram_npz['cuts'][:cutoff]
    noflips_th_arrivals = seismogram_npz['theory'][:cutoff]
    flips_cut_times = seismogram_npz['cuts'][cutoff:]
    flips_th_arrivals = seismogram_npz['theory'][cutoff:]
    
    #pred = arrive_model.predict(np.load(file)).flatten()
    noflip_preds = model.predict(np.reshape(noflips, 
                                            (len(noflips), len(noflips[0]), 1))).flatten()
    flip_preds = model.predict(np.reshape(flips, 
                                         (len(flips), len(flips[0]), 1))).flatten()
    
    pred_means = [np.mean(noflip_preds + noflips_cut_times), 
                  np.mean(flip_preds + flips_cut_times)]
    pred_stds = [np.std(noflip_preds + noflips_cut_times), 
                 np.std(flip_preds + flips_cut_times)]
    
    if 0.2 > pred_stds[0]/pred_stds[1] > 5:
        correct_pred = np.argmin(pred_means)
    else:
        correct_pred = np.argmin(pred_stds)
    #th_arrival = noflips_th_arrivals[0] + noflips_cut_times[0]
    pred_time = pred_means[correct_pred]
    
    return pred_time, pred_stds[correct_pred], correct_pred

def best_Pred(seismogram_npz, models):
    n_models = len(models)
    model_preds = np.zeros(n_models)
    model_errs = np.zeros(n_models)
    model_flipped = np.zeros(n_models)
    for i, model in enumerate(models):
        pred_time, pred_std, flipped = make_Pred(model, seismogram_npz)
        model_preds[i] += pred_time
        model_errs[i] += pred_std
        model_flipped[i] += flipped
    
    best_model = np.argmin(model_errs)
    '''
    flip_votes = np.unique(model_flipped)[1]
    if np.unique(flip_votes).size < 2:
        flipped = 0
    else:
        flipped = np.argmax(flip_votes)
    '''
    return model_preds[best_model], model_errs[best_model], model_flipped[best_model]

def predict_Arrival(datadir, models, debug_mode=False):
    name = datadir.split('/')[-2]
    name = check_String(name)
    npzdir = './pred_data/' + name + '/'
    
    files = np.sort(os.listdir(npzdir))
    if debug_mode:
        files = files[:100]
    
    num_pred = len(files)
    seis_names = []
    pred_arrivals = np.zeros(num_pred)
    pred_uncertainties = np.zeros(num_pred)
    flipped = np.zeros(num_pred)
    for f, file in enumerate(files):
        file = check_String(file)
        seismogram = np.load(npzdir + file)
        pred_time, pred_std, flip = best_Pred(seismogram, models)
        '''
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
        '''
        seis_names.append(file.rstrip('.npz'))
        pred_arrivals[f] += pred_time
        pred_uncertainties[f] += pred_std
        flipped[f] += flip
        
    seis_names = np.asarray(seis_names)
    
    return seis_names, pred_arrivals, pred_uncertainties, flipped

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