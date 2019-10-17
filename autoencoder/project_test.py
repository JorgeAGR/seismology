#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 22:57:18 2019

@author: jorgeagr
"""

import numpy as np
import obspy
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, UpSampling1D, BatchNormalization, Reshape, Flatten
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam

''''
# This stumped me! I can't think recursively. Figure it out later.
def get_Decoder_Output(autoencoder, compression_size, layer, i=1, layers=12):
    if i == layers:
        layer = autoencoder.layers[-i](layer)
        decoder_input = Input(shape=(compression_size,))
        return decoder_input, layer(decoder_input)
    else:
        return get_Decoder_Output(autoencoder, compression_size, layer, )
'''
'''
    if layers >= i:
        layer = autoencoder.layers[-i](get_Decoder_Output(autoencoder, compression_size, i+1, layers=layers))
        return layer
    else:
        layer = autoencoder.layers[-i](get_Decoder_Output(autoencoder, compression_size, i+1, layers=layers))
        decoder_input = Input(shape=(compression_size,))
        return decoder_input, layer(decoder_input)
'''

def get_Callbacks(epochs):
    
    stopper = EarlyStopping(monitor='val_loss', min_delta=0.001, 
                            patience=epochs//2, restore_best_weights=True)
    checkpoint = ModelCheckpoint('./autoencoder_weights',
                                 monitor='val_loss', save_best_only=True, save_weights_only=True)
    return [stopper, checkpoint]

def get_Decoder_Layers(autoencoder, layers=12):
    
    #expand = autoencoder.layers[-12]
    decoder_layers = []
    for i in range(1, layers+1):
        decoder_layers.append(autoencoder.layers[-i])
        
    return decoder_layers

def rossNet_CAE(input_length, compression_size):
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

    flattened = Flatten()(max3)
    
    encoding = Dense(compression_size, activation='relu')(flattened)
    
    expand = Dense(max3.shape.as_list()[1] * max3.shape.as_list()[2], activation='relu')(encoding)
    
    reshaped = Reshape(max3.shape.as_list()[1:])(expand)
    
    convup1 = Conv1D(128, kernel_size=11, strides=1,
                     activation='relu', padding='same')(reshaped)
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
    
    autoencoder = Model(input_seis, decoding)
    
    decoder_input = Input(shape=(compression_size,))
    decoder_layers = get_Decoder_Layers(autoencoder)
    
    encoder = Model(input_seis, encoding)
    decoder = Model(decoder_input,
                    decoder_layers[0](decoder_layers[1](decoder_layers[2](decoder_layers[3](decoder_layers[4](decoder_layers[5](decoder_layers[6](decoder_layers[7](decoder_layers[8](decoder_layers[9](decoder_layers[10](decoder_layers[11](decoder_input)))))))))))))

    autoencoder.compile(loss='binary_crossentropy',#huber_loss,
                  optimizer=Adam(1e-3))

    return autoencoder, encoder, decoder

# Load training and testing data
batch_size = 128
epochs = 20
x_train = np.load('data/train/train_seismos.npy')
x_test = np.load('data/test/test_seismos.npy')

autoencoder, encoder, decoder = rossNet_CAE(x_train.shape[1], 32)
autoencoder.fit(x_train, x_train,
                validation_data=(x_test, x_test),
                batch_size=batch_size,
                epochs=epochs,
                callbacks=get_Callbacks(epochs))