#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 16:34:34 2019

@author: jorgeagr
"""
import os
# Safeguard for some systems
def check_String(string):
    return string.rstrip('\r')

def make_Dirs():
    directories = {'models': ['etc',], 'pred_data': None, 'results': None,
                   'src': None, 'train_data': ['etc',]}
    
    for key in directories:
        if key not in os.listdir('.'):
            os.mkdir(key)
        if type(directories[key]) == list:
            for subdir in directories[key]:
                if subdir not in os.listdir(key):
                    os.mkdir(key + '/' + subdir)

# Load cnn_config.txt file and reads the set values for variables to be used
def read_Config(file_path):
    config_dic = {}
    with open(file_path) as config:
        lines = config.readlines()
        for i, line in enumerate(lines):
            if (line != '\n') and ('#' not in line[0]):
                split_vals = False
                if ',' in line:
                    split_vals = True
                line = line.rstrip('\n').split(':')
                config_dic[line[0]] = [text for text in line[1:]]
                if split_vals:
                    config_dic[line[0]] = config_dic[line[0]][0].split(',')
                if (len(config_dic[line[0]]) == 1) and (line[0] != 'pred_dir'):
                    config_dic[line[0]] = config_dic[line[0]][0]
                try:
                    config_dic[line[0]] = int(config_dic[line[0]])
                except:
                    pass
    
    # Safeguard for some systems
    for key in config_dic:
        if type(config_dic[key]) == str:
            config_dic[key] = check_String(config_dic[key])
    
    return config_dic