#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 22:57:18 2019

@author: jorgeagr
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, UpSampling1D, BatchNormalization, Reshape, Flatten
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam, Adadelta

'''
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

def get_Callbacks(epochs, model_name):
    
    stopper = EarlyStopping(monitor='val_loss', min_delta=0.001, 
                            patience=epochs//2, restore_best_weights=True)
    checkpoint = ModelCheckpoint(model_name + '_weights',
                                 monitor='val_loss', save_best_only=True, save_weights_only=True)
    return [stopper, checkpoint]

def get_Decoder_Layers(autoencoder, layers=12):
    
    #expand = autoencoder.layers[-12]
    decoder_layers = []
    for i in range(1, layers+1):
        decoder_layers.append(autoencoder.layers[-i])
        
    return decoder_layers


def AutoEncoder(input_length, compression_size):
    input_seis = Input(shape=(input_length,))
    
    dense1 = Dense(input_length//2, activation='sigmoid')(input_seis)
    encoded = Dense(input_length//4, activation='sigmoid')(dense1)
    dense_up1 = Dense(input_length//2, activation='sigmoid')(encoded)
    decoded = Dense(input_length, activation='sigmoid')(dense_up1)

    autoencoder = Model(input_seis, decoded)
    autoencoder.compile(loss='mean_squared_error', optimizer=Adam(1e-4))

    return autoencoder, None, None

def RossNet_CAE(input_length, compression_size, transfer_learning=True):
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
    
    encoding = Dense(compression_size, activation='sigmoid')(flattened)
    
    expanded = Dense(max3.shape.as_list()[1] * max3.shape.as_list()[2], activation='relu')(encoding)
    
    reshaped = Reshape(max3.shape.as_list()[1:])(expanded)
    
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
    
    up1 = UpSampling1D(size=2)(reshaped)#(max3)
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
    # sigmoid? or tanh? or maybe something else
    decoding = Conv1D(1, kernel_size=21, strides=1,
                      activation='tanh', padding='same')(conv_up3)
    
    autoencoder = Model(input_seis, decoding)
    
    # Load weights from automated picker model for the initial feature recognition
    # Idea is that this should focus on reproducing the known shape of SS phase
    # both in main arrival and precursors
    if transfer_learning:
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
                  optimizer=Adam(1e-4))

    print(autoencoder.summary())

    return autoencoder, None, None#encoder, decoder

def train_Model(model_class, model_name, transfer_learning=True):
    # Load training and testing data
    batch_size = 128
    epochs = 20
    x_train = np.load('data/train/train_seismos.npy')
    x_test = np.load('data/test/test_seismos.npy')
    #x_train = (x_train + 1)/2
    #x_test = (x_test + 1)/2
    
    # For vanilla AE
    #x_train = x_train.reshape(x_train.shape[0], x_train.shape[1])
    #x_test = x_test.reshape(x_test.shape[0], x_test.shape[1])

    print(model_name)
    autoencoder, encoder, decoder = model_class(x_train.shape[1], 320, transfer_learning=transfer_learning)
    train_hist = autoencoder.fit(x_train, x_train,
                validation_data=(x_test, x_test),
                verbose=2,
                batch_size=batch_size,
                epochs=epochs,
                callbacks=get_Callbacks(epochs, model_name))
    autoencoder.load_weights(model_name + '_weights')
    autoencoder.save(model_name + '.h5')
    #encoder.save('encoder.h5')
    #decoder.save('decoder.h5')
    np.savez(model_name + '_train_log', loss=train_hist.history['loss'], val_loss=train_hist.history['val_loss'])

train_Model(RossNet_CAE, 'rossnet_convautoencoder_transfer_dense_mse')
#train_Model(RossNet_CAE, 'rossnet_convautoencoder_notransfer_nodense_mse', transfer_learning=False)
