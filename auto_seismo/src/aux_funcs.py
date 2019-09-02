#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 16:34:34 2019

@author: jorgeagr
"""

# Safeguard for some systems
def check_string(string):
    '''
    fixed_strings = []
    for string in strings:
        fixed_strings.append(string.rstrip('\r'))
    '''
    return string.rstrip('\r')

# Load cnn_config.txt file and reads the set values for variables to be used
def read_config(file_path):
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
            config_dic[key] = check_string(config_dic[key])
    
    return config_dic