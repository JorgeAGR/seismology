# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 22:28:30 2019

@author: jorge
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, UpSampling1D, BatchNormalization, Reshape, Flatten
from keras.models import Model, load_model
from keras.optimizers import Adam
import matplotlib as mpl

# Settings for plots
golden_ratio = (np.sqrt(5) + 1) / 2
width = 12
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


def RossNet_CAE(input_length, compression_size):
    input_seis = Input(shape=(input_length, 1))

    conv1 = Conv1D(32, kernel_size=21, strides=1,
                     activation='relu', padding='same')(input_seis)
    bn1 = BatchNormalization()(conv1)
    max1 = MaxPooling1D(pool_size=2)(bn1)

    conv2 = Conv1D(64, kernel_size=15, strides=1,
                     activation='relu', padding='same')(max1)
    bn2 = BatchNormalization()(conv2)
    max2 = MaxPooling1D(pool_size=2)(bn2)

    conv3 = Conv1D(128, kernel_size=11, strides=1,
                     activation='relu', padding='same')(max2)
    bn3 = BatchNormalization()(conv3)
    max3 = MaxPooling1D(pool_size=2)(bn3)

    #flattened = Flatten()(max3)
    
    #dense_d1 = Dense(500, activation='sigmoid')
    
    #encoding = Dense(compression_size, activation='sigmoid')(flattened)
    
    #expand = Dense(max3.shape.as_list()[1] * max3.shape.as_list()[2], activation='sigmoid')(encoding)
    
    #reshaped = Reshape(max3.shape.as_list()[1:])(expand)
    
    '''
    convup1 = Conv1D(128, kernel_size=11, strides=1,
                     activation='relu', padding='same')(max3)#(reshaped)
    bn_up1 = BatchNormalization()(convup1)
    up1 = UpSampling1D(size=2)(bn_up1)
    
    convup2 = Conv1D(64, kernel_size=15, strides=1,
                     activation='relu', padding='same')(up1)
    bn_up2 = BatchNormalization()(convup2)
    up2 = UpSampling1D(size=2)(bn_up2)
    
    convup3 = Conv1D(32, kernel_size=21, strides=1,
                     activation='relu', padding='same')(up2)
    bn_up3 = BatchNormalization()(convup3)
    up3 = UpSampling1D(size=2)(bn_up3)
    
    decoding = Conv1D(1, kernel_size=21, strides=1,
                     activation='sigmoid', padding='same')(up3)
    '''
    
    up1 = UpSampling1D(size=2)(max3)
    bn_up1 = BatchNormalization()(up1)
    conv_up1 = Conv1D(128, kernel_size=11, strides=1,
                     activation='relu', padding='same')(bn_up1)

    up2 = UpSampling1D(size=2)(conv_up1)
    bn_up2 = BatchNormalization()(up2)
    conv_up2 = Conv1D(64, kernel_size=15, strides=1,
                      activation='relu', padding='same')(bn_up2)

    up3 = UpSampling1D(size=2)(conv_up2)
    bn_up3 = BatchNormalization()(up3)
    conv_up3 = Conv1D(32, kernel_size=21, strides=1,
                      activation='relu', padding='same')(bn_up3)
    
    decoding = Conv1D(1, kernel_size=21, strides=1,
                      activation='sigmoid', padding='same')(conv_up3)
    
    autoencoder = Model(input_seis, decoding)
    
    # Load weights from automated picker model for the initial feature recognition
    # Idea is that this should focus on reproducing the known shape of SS phase
    # both in main arrival and precursors
    for layer, filt in zip((1, 4, 7), ('21', '15', '11')):
        autoencoder.layers[layer].set_weights(np.load('conv_weights/conv' + filt + 'x1.npy', allow_pickle=True))
        autoencoder.layers[layer].trainable = False
    #decoder_input = Input(shape=(compression_size,))
    #decoder_layers = get_Decoder_Layers(autoencoder)
    
    #encoder = Model(input_seis, max3)#encoding)
    #decoder_input = Input(shape=encoder.get_output_shape_at(0))
    #decoder = Model(decoder_input,
    #                decoder_layers[0](decoder_layers[1](decoder_layers[2](decoder_layers[3](decoder_layers[4](decoder_layers[5](decoder_layers[6](decoder_layers[7](decoder_layers[8](decoder_layers[9](decoder_layers[10](decoder_layers[11](decoder_input)))))))))))))
    
    # for losses either binary crossentropy or MSE
    autoencoder.compile(loss='mean_squared_error',
                  optimizer=Adam(1e-3))

    print(autoencoder.summary())

    return autoencoder, None, None#encoder, decoder

# Load training and testing data
x_train = np.load('data/train/train_seismos.npy')
x_test = np.load('data/test/test_seismos.npy')
# Shift to [0, 1] interval
# What the hell?! It works when training on [-1, 1] data, but predict with [0, 1]!?!??!
x_train = (x_train + 1)/2
x_test = (x_test + 1)/2 

# If using vanilla autoencoder
#x_train = x_train.reshape(x_train.shape[0], x_train.shape[1])
#x_test = x_test.reshape(x_test.shape[0], x_test.shape[1])

# Initialize models and load weights
autoencoder, encoder, decoder = RossNet_CAE(x_train.shape[1], 32)
autoencoder.load_weights('rossnet_convautoencoder_nodense_mse_weights')

#autoencoder = load_model('autoencoder_bce.h5')

# Predict for an instance and plot the actual and reconstructed for comparison
test_rec = autoencoder.predict(x_test[0].reshape(1, x_train.shape[1], 1)).flatten()
plt.plot(x_test[0])
plt.plot(test_rec)