# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 16:20:05 2019

@author: jorge
"""

import numpy
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import obspy

golden_ratio = (np.sqrt(5) + 1) / 2
width = 12
height = width / golden_ratio

mpl.rcParams['figure.figsize'] = (width, height)

mpl.rcParams['font.size'] = 28
mpl.rcParams['figure.titlesize'] = 'large'
mpl.rcParams['legend.fontsize'] = 'small'
mpl.rcParams['xtick.major.size'] = 12
mpl.rcParams['xtick.minor.size'] = 8
mpl.rcParams['xtick.labelsize'] = 24
mpl.rcParams['ytick.major.size'] = 12
mpl.rcParams['ytick.minor.size'] = 8
mpl.rcParams['ytick.labelsize'] = 24

datadir = '../seismograms/seis_2/'
files = ['20070130A.macquarie.W020.BHT.s_fil', '20070130A.macquarie.W12A.BHT.s_fil', '20070130A.macquarie.X13A.BHT.s_fil']

