#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 15:35:41 2018

@author: jorgeagr
"""
import os
import numpy as np
import keras
from keras.datasets import mnist
from keras.layers import Dense, Flatten
from keras.layers import Conv1D, MaxPooling1D
from keras.models import Sequential

datadir = '.'
'''
files = os.listdir(datadir)

seismograms = []
arrivals = []
for file in files:
    seismogram = obspy.read(datadir + file)
    N = seismogram[0].stats.npts
    delta = np.round(seismogram[0].stats.delta, 1)
    b = seismogram[0].stats.sac['b']
    e = seismogram[0].stats.sac['e']
    shift = -b

    amp = seismogram[0].data
    time = np.arange(b + shift, e + delta + shift, delta)

    arrival = seismogram[0].stats.sac['t6'] + shift
    arrival_approx = np.round(arrival)
    arrival_ind = np.argwhere(np.isclose(arrival, time, atol=0.1))[0][0]
    init, end = int(arrival_ind - 20/delta), int(arrival_ind + 20/delta)
    time = time[init:end]
    arrival += -time[0]
    time += -time[0]
    amp = amp[init:end]
    seismograms.append(amp)
    arrivals.append(arrival)

    fig, ax = plt.subplots()
    fig.set_size_inches(10, 5)
    ax.plot(time, amp)
    ax.vlines(arrival, ymin=min(amp), ymax=max(amp)*1.1, color='red')
    ax.set_xlim(time[0], time[-1])
    plt.show()
    break

seismograms = np.array(seismograms)
seismograms = seismograms.reshape(len(seismograms), len(seismograms[0]), 1)
arrivals = np.array(arrivals)
arrivals = arrivals.reshape(len(arrivals), 1)
'''

seismograms = np.load('seismograms.npy')
arrivals = np.load('arrivals.npy')

test_sample = 10000
cutoff = len(seismograms) - test_sample

train_x = seismograms[:cutoff]
train_y = arrivals[:cutoff]

test_x = seismograms[cutoff:]
test_y = arrivals[cutoff:]

model = Sequential()
model.add(Conv1D(32, kernel_size=5, strides=1,
                 activation='relu',
                 input_shape=(len(seismograms[0]), 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(64, 5, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(1))

model.compile(loss='mse',
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

model.fit(test_x, test_y,
          batch_size=16,
          epochs=20,
          verbose=1,)
          #validation_data=(test_x, test_y),)
          #callbacks=[history])
pred = model.predict(seismograms)
print(model.evaluate(test_x, test_y))