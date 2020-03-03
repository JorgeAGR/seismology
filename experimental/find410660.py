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
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from scipy.stats import norm

# Settings for plots
golden_ratio = (np.sqrt(5) + 1) / 2
width = 15
height = width / golden_ratio

mpl.rcParams['figure.figsize'] = (width, height)
mpl.rcParams['font.size'] = 14
mpl.rcParams['figure.titlesize'] = 'large'
mpl.rcParams['legend.fontsize'] = 'small'
mpl.rcParams['xtick.major.size'] = 12
mpl.rcParams['xtick.minor.size'] = 8
mpl.rcParams['xtick.labelsize'] = 14
mpl.rcParams['ytick.major.size'] = 12
mpl.rcParams['ytick.minor.size'] = 8
mpl.rcParams['ytick.labelsize'] = 14

df = pd.read_csv('cross_secs_dat/model_pred/5caps_deg_preds.csv')
pred_inds = np.asarray([1, 5, 9, 13, 17])
err_inds = pred_inds + 1
amp_inds = err_inds + 1
qual_inds = amp_inds + 1

arrivals = df.values[:,pred_inds].flatten()
errors = df.values[:,err_inds]
amps = df.values[:,amp_inds]
qualities = df.values[:,qual_inds]

eps = 4
percent = 0.99
dbscan = DBSCAN(eps=eps, min_samples=np.round(len(df)*percent))
dbscan.fit(arrivals.reshape(-1,1))



sigma = 2
'''
This is to visualize a statistical interpretation of how the predictions will
be clustered together. Assuming normally distributed clusters of predictions,
eps can be considered the number of standard deviations that correspond to
percent. So that for 95% of the data, eps is 2sigma. So given eps = 5 and 
percent = 0.95, this can be translated as saying that for a data point to be 
considered a core point (a mean point, say), then at least 95% of the predictions
should be within 2sigma (2sigma = 3 units away from it).

ONLY an approximate interpretation though! Since for a single sample, there can be
multiple predictions within a cluster (say -153, -155, -157 for the 410) with
different quality levels. This generates more than the expected number of data
points, cuasing points to be considered outside of the given confidence level.
Although, regardless of this they still seem to exist within the entire normal
distribution at least, which is good.
'''
core_points = dbscan.core_sample_indices_
core0_mu = arrivals[core_points][dbscan.labels_[core_points] == 0].mean()
core1_mu = arrivals[core_points][dbscan.labels_[core_points] == 1].mean()
cluster0 = np.where(dbscan.labels_ == 0)
cluster1 = np.where(dbscan.labels_ == 1)
arrival_ind_grid = np.arange(len(arrivals))
x_grid = np.linspace(min(arrivals), max(arrivals), num=200)
fig, ax = plt.subplots()
ax.plot(arrivals, np.arange(len(arrivals)) / len(arrivals), 'k.')
for c in [cluster0, cluster1]:
    ax.plot(arrivals[c], arrival_ind_grid[c] / len(arrivals), '.', color='orange')
ax.plot(arrivals[core_points], arrival_ind_grid[core_points] / len(arrivals), 'r.')
for mu in [core0_mu, core1_mu]:
    normal = norm.pdf(x_grid, loc=mu, scale=eps/sigma)+0.5
    ax.plot(x_grid, normal, color='green')
    ax.axvline(mu-eps, color='green')
    ax.axvline(mu+eps, color='green')
    #ax.fill_betweenx(normal[normal >= twosigma][:-1], np.arange(mu-2*eps,mu+2*eps+1), facecolor='green')
fig.tight_layout(pad=0.5)



clusters, freq = np.unique(dbscan.labels_, return_counts=True)
#clusters, freq = np.unique(gmm.labels_, return_counts=True)
if -1 in clusters:
    clusters = clusters[1:]
    freq = freq[1:]
# Keep the following?
global_ind = np.argsort(freq)[-2:]
clusters, freq = clusters[global_ind], freq[global_ind]
avg_preds = np.asarray([arrivals[dbscan.labels_ == c].mean() for c in clusters])
sort_ind = np.argsort(avg_preds)
cluster660, cluster410 = clusters[sort_ind]

labels = dbscan.labels_.reshape(len(df), len(pred_inds))

ind410 = np.zeros(len(df), dtype=np.int)
ind660 = np.zeros(len(df), dtype=np.int)

for i in range(len(labels)):
    l, counts = np.unique(labels[i], return_counts=True)
    if (cluster410 not in l) or (cluster660 not in l):
        labels[i] = -np.ones(len(labels[i]))
        ind410[i] = -1
        ind660[i] = -1
        continue
    for c in clusters:
        if counts[l==c][0] > 1:
            where = np.argwhere(labels[i]==c).flatten()
            maxqual = np.argmax(qualities[i][where])
            throw = [j for j in range(len(where)) if j != maxqual]
            labels[i][where[throw]] = -1
    ind410[i] = np.where(labels[i]==cluster410)[0][0]
    ind660[i] = np.where(labels[i]==cluster660)[0][0]
    
arrivals = arrivals.reshape(len(df), len(pred_inds))

found410 = np.where(ind410 != -1)[0]
found660 = np.where(ind660 != -1)[0]
foundboth = np.union1d(found410, found660)
ind410 = ind410[foundboth]
ind660 = ind660[foundboth]

preds410, preds660 = arrivals[foundboth,ind410], arrivals[foundboth,ind660]
errs410, errs660 = errors[foundboth,ind410], errors[foundboth,ind660]
amps410, amps660 = amps[foundboth,ind410], amps[foundboth,ind660]
quals410,quals660 = qualities[foundboth,ind410], qualities[foundboth,ind660]

df_new = pd.DataFrame(data = {'file':df['file'].values[foundboth],
                              '410pred':preds410, '410err':errs410, '410amp':amps410, '410qual':quals410,
                              '660pred':preds660, '660err':errs660, '660amp':amps660, '660qual':quals660})
    
print('Num:', df_new.shape[0], '/', df.shape[0])
print('Min:', df_new['410pred'].min(), df_new['660pred'].min())
print('Max:', df_new['410pred'].max(), df_new['660pred'].max())