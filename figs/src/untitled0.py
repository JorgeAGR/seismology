#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 15:05:01 2020

@author: jorgeagr
"""

import numpy as np
import obspy
import pandas as pd
import os

entries = {}
for file in os.listdir('/home/jorgeagr/Documents/seismograms/SS_picked/'):
    if '.sac' in file:
        seis = obspy.read('/home/jorgeagr/Documents/seismograms/SS_picked/' + file)
        time = seis[0].stats.sac.t2
        qual = seis[0].stats.sac.user2
        entries[file.rstrip('_auto.sac')] = (time, qual)

with open('/home/jorgeagr/Documents/seismograms/ss_picks.csv', 'w+') as csv:
    print('file,pick,qual', file=csv)
    for key in entries:
        print('{},{:.2f},{:.2f}'.format(key, entries[key][0], entries[key][1]), file=csv)