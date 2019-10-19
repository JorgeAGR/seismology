# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 22:28:30 2019

@author: jorge
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, UpSampling1D, BatchNormalization, Reshape, Flatten
from keras.models import Model
from keras.optimizers import Adam

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

    # for losses either binary crossentropy or MSE
    autoencoder.compile(loss='mean_squared_error',#huber_loss,
                  optimizer=Adam(1e-3))

    return autoencoder, encoder, decoder

# Load training and testing data
batch_size = 128
epochs = 20
x_train = np.load('data/train/train_seismos.npy')
x_test = np.load('data/test/test_seismos.npy')
# Shift to [0, 1] interval
x_train = (x_train + 1)/2
x_test = (x_test + 1)/2 

# Initialize models and load weights
autoencoder, encoder, decoder = rossNet_CAE(x_train.shape[1], 32)
autoencoder.load_weights('autoencoder_weights')

# Predict for an instance and plot the actual and reconstructed for comparison
train_rec = autoencoder.predict(x_train[0].reshape(1, x_train.shape[1], 1)).flatten()
plt.plot(x_train[0])
plt.plot(train_rec)
