# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 14:23:29 2019

@author: jorge
"""
import os
import numpy as np
import seismo_arrays

config_dic = {}
with open('./cnn_config.txt') as config:
    lines = config.readlines()
    for i, line in enumerate(lines):
        if (line != '\n') and ('#' not in line):
            split_vals = False
            if '||' in line:
                split_vals = True
            line = line.rstrip('\n').split(':')
            config_dic[line[0]] = [text for text in line[1:]]
            if split_vals:
                config_dic[line[0]] = config_dic[line[0]][0].split('||')
            if (len(config_dic[line[0]]) == 1) and (line[0] != 'pred_dir'):
                config_dic[line[0]] = config_dic[line[0]][0]
            try:
                config_dic[line[0]] = int(config_dic[line[0]])
            except:
                pass
            
for d in config_dic['pred_dir']:
    #seismo_arrays.make_arrays(d, config_dic['arrival_var'])
    files = os.listdir(d)
    np.save('./results/file_names_' + d.split('/')[-2], np.asarray(files))
    #with open('./results/file_names_' + d.split('/')[-2], 'w+') as file:
    #    for line in files:
    #        print(line, file=file)