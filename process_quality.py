#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 22:35:00 2018

@author: jorgeagr
"""

import os
import obspy
import numpy as np

datadir_good = './SS_kept/'
datadir_bad = './SS_bad/'

def process(datadir, arrival_var):
    try:
        name = datadir.split('/')[1]
        
        files = os.listdir(datadir)
        
        seismograms = []
        arrivals = []
        #arrivals_theory = []
        for i, file in enumerate(files):
            print(i,'/', len(files))
            seismogram = obspy.read(datadir + file)
            N = seismogram[0].stats.npts
            delta = np.round(seismogram[0].stats.delta, 1)
            b = seismogram[0].stats.sac['b']
            e = seismogram[0].stats.sac['e']
            shift = -b
            if b > 0:
                continue
            elif e < 0:
                continue
            
            amp = seismogram[0].data
            time = np.arange(b + shift, e + delta + shift, delta)
            if len(time) > len(amp):
                time = time[:len(amp)]
            
            arrival = seismogram[0].stats.sac[arrival_var] + shift
            #theoretical_arrival = seismogram[0].stats.sac['t2'] + shift
            
            del seismogram
            
            rand_window = np.random.rand(5)
            for n in rand_window:
                #if arrival > time[-1]:
                #    continue
                init = np.where(arrival-20-15*n < time)[0][0]
                end = np.where(arrival+20-15*n > time)[0][-1]
                while True:
                    if end-init < 400:
                        end += 1
                    elif end-init > 400:
                        init += 1
                    else:
                        break
                #arrival_approx = np.round(arrival)
                #arrival_ind = np.argwhere(np.isclose(arrival, time, atol=0.1))[0][0]
                #init, end = int(arrival_ind - 20/delta), int(arrival_ind + 20/delta)
                
                #amp = amp[np.where(arrival-20 < time)]S
                #print(len(amp))
                #time = time[np.where(arrival-20 < time)]
                #amp = amp[np.where(time < arrival+20)].tolist()
                #time = time[np.where(time < arrival+20)]
                
                #time = time[init:end]
                #arrival += -time[0]
                #time += -time[0]
                
                amp_i = amp[init:end]
                seismograms.append(amp_i)
                arrivals.append(arrival)
                #arrivals_theory.append(theoretical_arrival)
            
            
        seismograms = np.array(seismograms)
        seismograms = seismograms.reshape(len(seismograms), len(seismograms[0]), 1)
        arrivals = np.array(arrivals)
        arrivals = arrivals.reshape(len(arrivals), 1)
        
        #arrivals_theory = np.array(arrivals_theory)
        #arrivals_theory = arrivals_theory.reshape(len(arrivals_theory), 1)
        
        np.save('seismograms_' + name, seismograms)
        np.save('arrivals_' + name, arrivals)
    except:
        seismograms = np.array(seismograms)
        seismograms = seismograms.reshape(len(seismograms), len(seismograms[0]), 1)
        arrivals = np.array(arrivals)
        arrivals = arrivals.reshape(len(arrivals), 1)
        #arrivals_theory = np.array(arrivals_theory)
        #arrivals_theory = arrivals_theory.reshape(len(arrivals_theory), 1)
        
        np.save('seismograms_' + name, seismograms)
        np.save('arrivals_' + name, arrivals)
        
#np.save('arrivals_theory', arrivals)

#process(datadir_good, 't6')
process(datadir_bad, 't2')

# Stuff for plotting. Plug in when necessary.
'''
    amps.append(amp[init:end])
    ts.append(time[init:end])
    
fig, ax = plt.subplots(nrows=5)
#fig.set_size_inches(10, 5)
for i, a in enumerate(ax):
    a.plot(ts[i], amps[i], linewidth=1, color='black')
    a.axvline(arrival, linestyle='--', color='red')
    a.set_xlim(ts[i][0], ts[i][-1])
ax[-1].set_xlabel('Time [s]')
plt.tight_layout()
fig.savefig('test' + str(n) + '.png')
'''