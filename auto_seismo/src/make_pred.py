#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 14:22:23 2019

@author: jorgeagr
"""
import os
import numpy as np
from aux_funcs import check_String

def predict_Arrival(datadir, models, debug_mode=False, simple=False):
    '''
    Function that iterates through npz files of seismograms and obtains best prediction
    from various models for each file.
    
    Parameters
    ------------
    datadir : string
        Path to the seismogram files
    models : list of Keras Models
        Models to be used for predicting maximum arrival time
    debug_mode : boolean
        Shortens the number of predictions for debugging purposes
    simple : boolean
        If true, simply predicts off of a single unflipped signal instead of
        prediction from variations.
    
    Returns
    ------------
    seis_names : array of strings
        Array of the names of the files predicted for
    pred_arrivals : array of floats
        Array of the predicted maximum arrival times
    pred_uncertainties : array of floats
        Array of the uncertainty of the predicted maximum arrival times
    flipped : array of ints
        Array of wether the signal is to be flipped (1) or not (0)
    '''
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
        pred_time, pred_std, flip = best_Pred(seismogram, models, simple)
        seis_names.append(file.rstrip('.npz'))
        pred_arrivals[f] += pred_time
        pred_uncertainties[f] += pred_std
        flipped[f] += flip
    seis_names = np.asarray(seis_names)
    
    return seis_names, pred_arrivals, pred_uncertainties, flipped

def best_Pred(seismogram_npz, models, simple=False):
    '''
    Function that obtains predictions for seismograms using multiple models. Currently only
    returns prediction of model with lowest variance.
    
    Parameters
    ------------
    seismogram_npz : Numpy NpzFile
    
    models : list of Keras Models
    
    simple : boolean
    
    Returns
    ------------
    model_preds : array of floats
    
    model_errs : array of floats
    
    model_flipped : array of ints
    
    '''
    n_models = len(models)
    model_preds = np.zeros(n_models)
    model_errs = np.zeros(n_models)
    model_flipped = np.zeros(n_models)
    for i, model in enumerate(models):
        if simple:
            pred_time = model.predict(seismogram_npz['noflips'][0].reshape(1,400,1))
            pred_std, flipped = 0, 0
        else:
            pred_time, pred_std, flipped = make_Pred(model, seismogram_npz)
        model_preds[i] += pred_time
        model_errs[i] += pred_std
        model_flipped[i] += flipped
    if simple:
        best_model = 0
    else:    
        best_model = np.argmin(model_errs)
    
    return model_preds[best_model], model_errs[best_model], model_flipped[best_model]

def make_Pred(model, seismogram_npz):
    '''
    Function that runs the flipped and unflipped signals for a single seismogram
    through the model and predicts for them. Allows to obtain an uncertainty in
    the prediction, as well as determine the correct polarity from the lower 
    prediction variance.
    
    Parameters
    ------------
    model : Keras Model
    
    seismogram_npz : Numpy NpzFile
    
    Returns
    ------------
    pred_time : float
    
    pred_std : float
    
    correct_pred : int
    '''
    noflips = seismogram_npz['noflips']
    flips = seismogram_npz['flips']
    cutoff = len(noflips)
    noflips_cut_times = seismogram_npz['cuts'][:cutoff]
    #noflips_th_arrivals = seismogram_npz['theory'][:cutoff]
    flips_cut_times = seismogram_npz['cuts'][cutoff:]
    #flips_th_arrivals = seismogram_npz['theory'][cutoff:]
    
    noflip_preds = model.predict(np.reshape(noflips, 
                                            (len(noflips), len(noflips[0]), 1))).flatten()
    flip_preds = model.predict(np.reshape(flips, 
                                         (len(flips), len(flips[0]), 1))).flatten()
    
    pred_means = [np.mean(noflip_preds + noflips_cut_times), 
                  np.mean(flip_preds + flips_cut_times)]
    pred_stds = [np.std(noflip_preds + noflips_cut_times), 
                 np.std(flip_preds + flips_cut_times)]
    
    if 0.5 > pred_stds[0]/pred_stds[1] > 2:
        correct_pred = np.argmin(pred_means)
    else:
        correct_pred = np.argmin(pred_stds)
    pred_time = pred_means[correct_pred]
    
    return pred_time, pred_stds[correct_pred], correct_pred