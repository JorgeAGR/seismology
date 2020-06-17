#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 21:12:53 2020

@author: jorgeagr
"""

import numpy as np
import obspy
import os
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='Read SAC files in a directory and write picks to CSV.')
parser.add_argument('sac_dir', help='Directory of the SAC files', type=str)
parser.add_argument('ext', help='File extension', type=str)
parser.add_argument('auto_var', help='Variable for auto pick', type=str)
parser.add_argument('qual_var', help='Variable for pick quality', type=str)
args = parser.parse_args()

file_dir = args.sac_dir
ext = args.ext
ap = args.auto_var
q = args.qual_var

if file_dir[-1] != '/':
    file_dir = file_dir + '/' 
if ext[0] != '.':
    ext = '.'+ext

files = [f for f in os.listdir(file_dir) if ext in f]

results = {}
for f in files:
    seis = obspy.read(file_dir + f)
    auto = seis[0].stats.sac[ap]
    qual = seis[0].stats.sac[q]
    results[f] = (auto, qual)
    
with open('{}../{}_picks.csv'.format(file_dir, file_dir.split('/')[-2]), 'w+') as csv:
    print('file,prediction,quality', file=csv)
    for key in results:
        name = key.rstrip(ext)
        if '_auto' in name:
            name = name.rstrip('_auto')
        print('{},{:.2f},{:.2f}'.format(key, results[key][0], results[key][1]), file=csv)