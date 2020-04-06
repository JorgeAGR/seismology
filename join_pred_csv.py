#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 15:40:30 2020

@author: jorgeagr
"""

import os
import argparse
import pandas as pd

parser = argparse.ArgumentParser(description='Join prediction CSVs from a multijob prediction.')
parser.add_argument('file_path', help='Path to the files', type=str)
parser.add_argument('data_name', help='Name of data set that files follow (NameOfFile#_preds.csv)', type=str)
args = parser.parse_args()

file_path = args.file_path
data_name = args.data_name

if file_path[-1] != '/':
    file_path += '/'

files = sorted([file for file in os.listdir(file_path) if data_name in file])

df = pd.read_csv(file_path + files[0])
for i in range(1, len(files)):
    df = df.append(pd.read_csv(file_path + files[i]))
    
df.to_csv(file_path + data_name + '_preds.csv', index=False)