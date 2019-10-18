# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 23:31:49 2019

@author: jorge
"""
import os
import obspy
import numpy as np
import scipy as sp
from aux_funcs import check_String

def write_Pred(datadir, files, preds, flipped, pred_var):
    '''
    Function that writes out the predicted arrival times to SAC files
    datadir
    
    Parameters
    ------------
    datadir : string
        String of the directory where the files are located
    files : array/list
        List of files for which predictions will be written
    preds : array/list
        Predicted arrival times from the model used
    flipped : array/list
        If a seismogram has positive or negative polarity
    pred_var : string
        Name of header where the predicted arrival time will be written
    
    Returns
    ------------
    N/A
    '''
    # Iterate through filers in a given file list and path directory
    for i, file in enumerate(files):
        # Load SAC file
        seismogram = obspy.read(datadir + file + '.s_fil')
        # Stor prediction in prediction variable header
        seismogram[0].stats.sac[pred_var] = preds[i]
        # Flip the seismogram to positive polarity if needed
        if flipped[i]:
            None
            #seismogram[0].data = -seismogram[0].data
        # Shift prediction to its closest maximum
        #seismogram[0].stats.sac[pred_var] = shift_Max(seismogram[0], pred_var)
        # Write out the file with the new header containing the prediction
        seismogram.write(datadir + file + '.s_fil', format='SAC')
    
    return

def shift_Max(seis, pred_var):
    '''
    Function to shift the model's prediction to the closest local maximum
    in the signal.
    
    Parameters
    ------------
    seis : obspy Trace
        Trace type from Obspy when loading a SAC file
    pred_var : string
        Name of header where the predicted arrival time is located
    
    Returns
    ------------
    arrival : float
        The closest local maximum to which the prediction should be shifted
    '''
    data = seis.data
    # Generate an time array for the data
    time = seis.times()
    # Initialize arrival variables to measure change
    arrival = 0
    new_arrival = seis.stats.sac[pred_var]
    while (new_arrival - arrival) != 0:
        arrival = new_arrival
        # Find indices of times 1 second before and after the current arrival
        init = np.where(time > (arrival - 1))[0][0]
        end = np.where(time > (arrival + 1))[0][0]
        # Find index of local maximum amplitude in previous time interval
        amp_max = np.argmax(np.abs(data[init:end]))
        # Change index in terms to total time range, not just 2 sec window
        time_ind = np.arange(init, end, 1)[amp_max]
        # Update arrival
        new_arrival = time[time_ind]
    return arrival
