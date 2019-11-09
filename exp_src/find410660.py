#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 16:07:19 2019

@author: jorgeagr
"""

import numpy as np
import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from sklearn.cluster import DBSCAN

df = pd.read_csv('15caps_wig_preds.csv')
pred_inds = np.asarray([1, 5, 9, 13, 17])
err_inds = pred_inds + 1
amp_inds = err_inds + 1
qual_inds = amp_inds + 1

arrivals = df.values[:,pred_inds].flatten()
errors = df.values[:,err_inds]
amps = df.values[:,amp_inds]
qualities = df.values[:,qual_inds]

dbscan = DBSCAN(eps=5, min_samples=len(df)*9//10)
dbscan.fit(arrivals.reshape(-1,1))

clusters = np.unique(dbscan.labels_)
if -1 in clusters:
    clusters = clusters[1:]
avg_preds = np.asarray([arrivals[dbscan.labels_ == c].mean() for c in clusters])
sort_ind = np.argsort(avg_preds)
cluster660, cluster410 = clusters[sort_ind]

labels = dbscan.labels_.reshape(len(df), len(pred_inds))

ind410 = np.zeros(len(df), dtype=np.int)
ind660 = np.zeros(len(df), dtype=np.int)
'''
Need to implement removal of predictions that did not find either discontinuity
'''
for i in range(len(labels)):
    l, counts = np.unique(labels[i], return_counts=True)
    for c in clusters:
        if counts[l==c][0] > 1:
            where = np.argwhere(labels[i]==c).flatten()
            maxqual = np.argmax(qualities[i][where])
            labels[i][where[~maxqual]] = -1
    ind410[i] = np.where(labels[i]==cluster410)[0][0]
    ind660[i] = np.where(labels[i]==cluster660)[0][0]
    
arrivals = arrivals.reshape(len(df), len(pred_inds))

preds410, preds660 = arrivals[range(len(df)),ind410], arrivals[range(len(df)),ind660]
 