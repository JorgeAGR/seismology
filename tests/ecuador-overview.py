#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 17:16:38 2018

@author: jorgeagr
"""

import os
import obspy
import numpy as np
import pandas as pd

name = []
s = []
scs = []
deltas = []

datadir = '../2.picking/20150327.ecuador/bht/filt/'

for file in os.listdir(datadir):
    if '.s_fil' in file:
        seismogram = obspy.read(datadir + file)
        if ('t2' in seismogram[0].stats.sac.keys()) and ('t3' in seismogram[0].stats.sac.keys()):
            name.append(file.split('.')[1])
            s.append(seismogram[0].stats.sac['t2'] - seismogram[0].stats.sac['b'])
            scs.append(seismogram[0].stats.sac['t3'] - seismogram[0].stats.sac['b'])
            deltas.append(seismogram[0].stats.sac['delta'])

name = np.asarray(name)
s = np.asarray(s)
scs = np.asarray(scs)
deltas = np.asarray(deltas)

indices = np.where(deltas == 0.025)

name = name[indices]
s = s[indices]
scs = scs[indices]
deltas = deltas[indices]

df = pd.DataFrame(data = {'ID': name, 'S': s, 'SCS': scs, 'Delta': deltas})