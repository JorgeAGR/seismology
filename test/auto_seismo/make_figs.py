# -*- coding: utf-8 -*-
"""
Created on Sun May 26 17:24:23 2019

@author: jorge
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import obspy

golden_ratio = (np.sqrt(5) + 1) / 2
width = 12
height = width / golden_ratio

mpl.rcParams['figure.figsize'] = (width, height)

seismograms = np.load('train_data/seismograms_t2.npy')
#arrive_model = load_model('./models/pred_model_2_norm.h5')
cutoff = int(len(seismograms) - len(seismograms)*0.15)
pred = arrive_model.predict(seismograms).flatten()
arrivals = np.load('train_data/arrivals_t2.npy').flatten()
time_i = np.load('train_data/cut_times_t2.npy')
error = pred[cutoff:] - arrivals[cutoff:]
abs_error_mean = np.mean(np.abs(error))
abs_error_std = np.std(np.abs(error))

mpl.rcParams['font.size'] = 28
mpl.rcParams['figure.titlesize'] = 'large'
mpl.rcParams['legend.fontsize'] = 'small'
mpl.rcParams['xtick.major.size'] = 12
mpl.rcParams['xtick.minor.size'] = 8
mpl.rcParams['xtick.labelsize'] = 24
mpl.rcParams['ytick.major.size'] = 12
mpl.rcParams['ytick.minor.size'] = 8
mpl.rcParams['ytick.labelsize'] = 24

datadir = '../../seismology/seismos/SS_kept/'
files = os.listdir(datadir)
arrival_diff = []
for i, file in enumerate(files):
    print('File', i+1, '/', len(files), '...')
    seismogram = obspy.read(datadir + file)[0]
    b = seismogram.stats.sac['b']
    shift = -b
    th_arrival = seismogram.stats.sac['t2'] + shift
    pick_arrival = seismogram.stats.sac['t6'] + shift
    arrival_diff.append(th_arrival - pick_arrival)
arrival_diff = np.asarray(arrival_diff)
fig, ax = plt.subplots()
weights = np.ones_like(arrival_diff)/len(arrival_diff)
hist = ax.hist(np.abs(arrival_diff), np.arange(0, 50, 0.01), histtype='step', align='mid', 
        color='black', linewidth=1, weights=weights, cumulative=True)
ax.axvline(30, color='black', linestyle='--')
ax.set_xlim(0.1, 35)
ax.xaxis.set_major_locator(mtick.MultipleLocator(5))
ax.xaxis.set_minor_locator(mtick.MultipleLocator(1))
ax.yaxis.set_minor_locator(mtick.MultipleLocator(0.05))
ax.set_ylabel('Fraction of Total Samples')
ax.set_xlabel(r'$|t_{theory} - t_{true}|$ [s]')
plt.tight_layout()
plt.savefig('data_cum.svg')


fig, ax = plt.subplots()
weights = np.ones_like(error)/len(error)
hist = ax.hist(error, np.arange(-10, 10, 0.05), histtype='stepfilled', align='mid', 
        color='black', linewidth=1, weights=weights, cumulative=False)
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(0, 0.11)
ax.xaxis.set_major_locator(mtick.MultipleLocator(0.5))
ax.xaxis.set_minor_locator(mtick.MultipleLocator(0.1))
ax.yaxis.set_minor_locator(mtick.MultipleLocator(0.01))
ax.set_ylabel('Fraction of Total Samples')
ax.set_xlabel(r'$t_{theory} - t_{true}$ [s]')
plt.tight_layout()
plt.savefig('pred_hist.svg')

fig, ax = plt.subplots()
weights = np.ones_like(error)/len(error)
hist = ax.hist(np.abs(error), np.arange(-10, 10, 0.002), histtype='step', align='mid', 
        color='black', linewidth=1, weights=weights, cumulative=True)
ax.set_xlim(0, 2)
ax.xaxis.set_major_locator(mtick.MultipleLocator(0.5))
ax.xaxis.set_minor_locator(mtick.MultipleLocator(0.1))
ax.set_ylim(0, 1)
ax.yaxis.set_major_locator(mtick.MultipleLocator(0.1))
ax.yaxis.set_minor_locator(mtick.MultipleLocator(0.02))
ax.set_ylabel('Fraction of Total Samples')
ax.set_xlabel(r'$|t_{pred} - t_{true}|$ [s]')
#ax.grid()
plt.tight_layout()
plt.savefig('pred_cum.svg')

mpl.rcParams['font.size'] = 18
mpl.rcParams['figure.titlesize'] = 'large'
mpl.rcParams['legend.fontsize'] = 'small'
mpl.rcParams['xtick.major.size'] = 8
mpl.rcParams['xtick.minor.size'] = 4
mpl.rcParams['xtick.labelsize'] = 14
mpl.rcParams['ytick.major.size'] = 8
mpl.rcParams['ytick.minor.size'] = 4
mpl.rcParams['ytick.labelsize'] = 14

fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True)
#sample_seis = np.random.randint(0, high=len(seismograms), size=4)
#sample_seis = np.array([34982, 197662, 22390, 165919])
# good one: 34982, 197662, 22390, 165919
sample_seis = np.array([0, 1, 2, 3])
#t = np.arange(0, 40, 0.1)
for n, i in enumerate(sample_seis):
    s = obspy.read(datadir+files[i])[0]
    t = s.times()
    init = np.where(t > s.stats.sac.t2 - 10 - s.stats.sac.b)[0][0]
    end = init + 1000
    t = t[init:end] - t[init]
    amp_i = s.data[init:end]
    amp_i = (amp_i - amp_i.min()) / (amp_i.max() - amp_i.min())
    p = arrive_model.predict(amp_i[:400].reshape((1, 400, 1))).flatten()[0]
    a = s.stats.sac.t6 - s.stats.sac.t2 + 10
    ax[n//2][n%2].plot(t, amp_i, color='black')
    ax[n//2][n%2].axvline(p, color='black', linestyle='--', label='Prediction')
    ax[n//2][n%2].axvline(a, color='red', label='Actual')
    ax[n//2][n%2].set_xlim(0, 100)
    ax[n//2][n%2].set_ylim(-0.05, 1.05)
    ax[n//2][n%2].xaxis.set_minor_locator(mtick.MultipleLocator(5))
    ax[n//2][n%2].yaxis.set_minor_locator(mtick.MultipleLocator(0.1))
    #ax[1][n%2].set_xlabel('Time From Cut [s]')
    #ax[n//2][n%1].set_ylabel('Relative Amplitude')
ax[0][1].legend(fontsize=12)
fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.xlabel('Time From Cut [s]')
plt.ylabel('Relative Amplitude')
#fig.text(0.5, 0.02, 'Time From Cut [s]', ha='center')
#fig.text(0.04, 0.5, 'Relative Amplitude', va='center', rotation='vertical')
plt.tight_layout()
plt.savefig('pred_eg.svg')
'''
'''
s_arr = np.load('./pred_data/seismograms_seis_1.npy')
t_i_arr = np.load('./pred_data/cut_times_seis_1.npy')
files = os.listdir('../seismograms/seis_1/')
fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True)
sample_seis = np.random.randint(0, high=len(files), size=4)
#t = np.arange(0, 40, 0.1)
for n, i in enumerate(sample_seis):
    s = obspy.read('../seismograms/seis_1/' + files[i])[0]
    p = arrive_model.predict(s_arr[i].reshape(1, len(s_arr[i]), 1))[0][0] + t_i_arr[i]
    ax[n//2][n%2].plot(s.times(), s.data)
    ax[n//2][n%2].axvline(p, color='black', linestyle='--', label='Prediction')
    ax[n//2][n%2].axvline(s.stats.sac.t6 - s.stats.sac.b, color='red', label='Picked')
    ax[1][n%2].set_xlabel('Time From Cut [s]')
    ax[n//2][n%1].set_ylabel('Relative Amplitude')
ax[0][1].legend()
plt.tight_layout()
plt.savefig('pred_egfull.svg')
'''
'''
# index = 22390//5 == 4478
train_dir = '../../seismology/seismos/SS_kept/'
train_files = os.listdir(train_dir)
seis_eg = obspy.read(train_dir + train_files[4479])[0]
init = 3850
end = 4250
#seis_eg.plot()
mpl.rcParams['figure.figsize'] = (12, 4)
fig, ax = plt.subplots()
ax.plot(seis_eg.times(), seis_eg.data, color='black')
ax.set_xlim(0, np.max(seis_eg.times()))
ax.set_ylim(-1.2, 1.1)
ax.xaxis.set_minor_locator(mtick.MultipleLocator(10))
ax.yaxis.set_major_locator(mtick.MultipleLocator(0.5))
ax.yaxis.set_minor_locator(mtick.MultipleLocator(0.1))
ax.axvline(init/10, color='black', linewidth=0.8)
ax.axvline(end/10, color='black', linewidth=0.8)
ax.set_xlabel('Time [s]')
ax.set_ylabel('Amplitude')
plt.tight_layout()
plt.savefig('seis_eg.svg')

seis_cut = seis_eg.data[init:end]
seis_minmax = (seis_cut - seis_cut.min()) / (seis_cut.max() - seis_cut.min())
pick = seis_eg.stats.sac.t6 - seis_eg.stats.sac.t2 + 10
mpl.rcParams['figure.figsize'] = (8, 4)
fig, ax = plt.subplots()
ax.plot(np.arange(0, 40, 0.1), seis_minmax, color='black')
ax.axvline(10, color='red', linestyle='--', label='Theoretical')
ax.axvline(pick, color='red', label='Actual')
ax.set_xlim(0, 40)
ax.set_ylim(0, 1)
plt.setp(ax.get_xticklabels(), visible=False)
#plt.setp(ax.get_yticklabels(), visible=False)
ax.tick_params(axis='x', which='both', length=0)
ax.yaxis.set_major_locator(mtick.MultipleLocator(1))
ax.legend()
plt.tight_layout()
plt.savefig('seis_eg_zoom.svg')