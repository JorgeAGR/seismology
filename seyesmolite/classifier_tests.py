#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 20:23:52 2020

@author: jorgeagr
"""

import os
from subprocess import call
import numpy as np
from tensorflow.keras.losses import Huber
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, BatchNormalization, Input, UpSampling1D, Reshape
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint
from time import time as clock
import obspy
from src.aux_funcs import check_String, read_Config

class SortingModel(object):
    
    def __init__(self, model_name):
        config = read_Config('models/{}.conf'.format(model_name))
        self.model_type = config['model_type']
        self.model_name = model_name
        self.model_path = 'models/{}/'.format(self.model_name)
        self.batch_size = config['batch_size']
        self.epochs = config['epochs']
        self.model_iters = config['model_iters']
        self.test_split = float(config['test_split'])
        self.debug = config['debug']
        
        self.files_path = list(map(lambda x: x+'/' if (x[-1] != '/') else x, [config['files_path'],]))[0]
        self.sample_rate = config['sample_rate']
        self.th_arrival_var = config['theory_arrival_var']
        self.arrival_var = config['pick_arrival_var']
        self.window_before = config['window_before']
        self.window_after = config['window_after']
        self.number_shift = config['number_shift']
        self.window_shift = config['window_shift']
        try:
            self.npz_path = config['temp_write_path'] + self.model_name
        except:
            self.npz_path = self.model_path
        
        self.total_time = (self.window_before + self.window_after) * self.sample_rate
        
        if self.model_name not in os.listdir('models/'):
            for directory in [self.model_path, self.model_path+'train_logs/']:
                os.mkdir(directory)
        return
    
    def __create_Train_Data(self):
        '''
        Function that iterates through seismograms in directory, perform preprocessing,
        create a time window around the arrival time and randomly shift it to augment
        the data set. Save augmented data set as npy files for quicker loading in
        the future. Meant for training/testing data.
        '''
        try:
            os.mkdir(self.npz_path+'npz/')
        except:
            pass
        files = np.sort(os.listdir(self.files_path))
        gen_whitespace = lambda x: ' '*len(x)
        
        for f, file in enumerate(files):
            if file+'npz' in os.listdir(self.npz_path+'npz/'):
                continue
            else:
                file = check_String(file)
                print_string = 'File ' + str(f+1) + ' / ' + str(len(files)) + '...'
                print('\r'+print_string, end=gen_whitespace(print_string))
                try:
                    seismogram = obspy.read(self.files_path + file)
                except:
                    continue
                seismogram = seismogram[0].resample(self.sample_rate)
                # Begging time
                b = seismogram.stats.sac['b']
                # The beginning time may not be 0, so shift all attribtues to be so
                shift = -b
                b = b + shift
                # End time
                e = seismogram.stats.sac['e'] + shift
                # Theoretical onset arrival time + shift
                if self.th_arrival_var == self.arrival_var:
                    th_arrival = seismogram.stats.sac[self.arrival_var] + shift - np.random.rand() * 20
                else:
                    th_arrival = seismogram.stats.sac[self.th_arrival_var] + shift
                # Picked maximum arrival time + shift
                arrival = seismogram.stats.sac[self.arrival_var] + shift
                
                # Theoretical arrival may be something unruly, so assign some random
                # shift from the picked arrival
                if not (b < th_arrival < e):
                    th_arrival = arrival - 20 * np.random.rand()
                
                amp = seismogram.data
                time = seismogram.times()
                # Shifts + 1 because we want a 0th shift + N random ones
                rand_window_shifts = 2*np.random.rand(self.number_shift+1) - 1 # [-1, 1] interval
                abs_sort = np.argsort(np.abs(rand_window_shifts))
                rand_window_shifts = rand_window_shifts[abs_sort]
                rand_window_shifts[0] = 0
                
                seis_windows = np.zeros((self.number_shift+1, self.total_time, 1))
                for i, n in enumerate(rand_window_shifts):
                    rand_arrival = th_arrival - n * self.window_shift
                    init = int(np.round((rand_arrival - self.window_before)*self.sample_rate))
                    end = init + self.total_time
                    if not (time[init] < arrival < time[end]):
                        init = int(np.round((arrival - 15 * np.random.rand() - self.window_before)*self.sample_rate))
                        end = init + self.total_time
                    amp_i = amp[init:end]
                    # Normalize by absolute peak, [-1, 1]
                    amp_i = amp_i / np.abs(amp_i).max()
                    seis_windows[i] = amp_i.reshape(self.total_time, 1)
                
                np.savez(self.npz_path+'npy/{}'.format(file), seis=seis_windows)
            
        return
    
    def __train_Test_Split(self, idnum, seed=None):
        npz_files = np.sort(os.listdir(self.npz_path+'npz/'.format(self.model_name)))
        cutoff = int(len(npz_files) * (1-self.test_split))
        np.random.seed(seed)
        np.random.shuffle(npz_files)
        train_npz_list = npz_files[:cutoff]
        test_npz_list = npz_files[cutoff:]
        np.savez(self.model_path+'train_logs/train_test_split{}'.format(idnum),
                 train=train_npz_list, test=test_npz_list)
        
        return train_npz_list, test_npz_list
    
    def __load_Data(self, npz_list, single=False):
        input_array = np.zeros((len(npz_list)*(self.number_shift+1)**(not single), self.total_time, 1))
        output_array = np.zeros((len(npz_list)*(self.number_shift+1)**(not single), self.total_time, 1))
        if single:
            for i, file in enumerate(npz_list):
                npz = np.load(self.npz_path+'npz/'+file)
                input_array[i] = npz['seis'][0]
                output_array[i] = npz['seis'][0]
        else:
            for i, file in enumerate(npz_list):
                npz = np.load(self.npz_path+'npz/'+file)
                input_array[(self.number_shift+1)*i:(self.number_shift+1)*(i+1)] = npz['seis']
                output_array[(self.number_shift+1)*i:(self.number_shift+1)*(i+1)] = npz['arrival']
        return input_array, output_array
    
    def __get_Callbacks(self, epochs):
        stopper = EarlyStopping(monitor='val_loss', min_delta=0, #don't want early stop 
                                patience=epochs, restore_best_weights=True)
        # Include Checkpoint? CSVLogger?
        return [stopper,]
    
    def train_Model(self):
        if self.debug:
            self.epochs=10
            self.model_iters=1
        
        self.__create_Train_Data()
        
        models = []
        models_train_means = np.zeros(self.model_iters)
        models_train_stds = np.zeros(self.model_iters)
        models_test_means = np.zeros(self.model_iters)
        models_test_stds = np.zeros(self.model_iters)
        models_test_final_loss = np.zeros(self.model_iters)
        
        models_train_lpe = np.zeros((self.model_iters, self.epochs))
        models_test_lpe = np.zeros((self.model_iters, self.epochs))
        tick = clock()
        for m in range(self.model_iters):        
            print('Training arrival prediction model', m+1)
            model = self.__rossNetAE()
            
            callbacks = self.__get_Callbacks(self.epochs)
            
            train_files, test_files = self.__train_Test_Split(m)
            train_x, train_y = self.__load_Data(train_files)
            test_x, test_y = self.__load_Data(test_files)
            
            train_hist = model.fit(train_x, train_y,
                                   validation_data=(test_x, test_y),
                                   batch_size=self.batch_size,
                                   epochs=self.epochs,
                                   verbose=2,
                                   callbacks=callbacks)
            
            total_epochs = len(train_hist.history['loss'])
            
            train_pred = model.predict(train_x)
            test_pred = model.predict(test_x)
            test_loss = model.evaluate(test_x, test_y,
                                       batch_size=self.batch_size, verbose=0)
            
            model_train_diff = np.abs(train_y - train_pred)
            model_test_diff = np.abs(test_y - test_pred)
            model_train_mean = np.mean(model_train_diff)
            model_train_std = np.std(model_train_diff)
            model_test_mean = np.mean(model_test_diff)
            model_test_std = np.std(model_test_diff)
            
            print('Train Error:{:.3f} +/- {:.3f}'.format(model_train_mean, model_train_std))
            print('Test Error:{:.3f} +/- {:.3f}'.format(model_test_mean, model_test_std))
            print('Test Loss:{:.3f}'.format(test_loss))
            
            models.append(model)
            models_train_means[m] += model_train_mean
            models_train_stds[m] += model_train_std
            models_test_means[m] += model_test_mean
            models_test_stds[m] += model_test_std
            models_test_final_loss[m] += test_loss
            models_train_lpe[m][:total_epochs] = train_hist.history['loss']
            models_test_lpe[m][:total_epochs] = train_hist.history['val_loss']
        
        #best_model = np.argmin(models_means)
        tock = clock()
        train_time = (tock-tick)/3600 # hours
        best_model = np.argmin(models_train_means)

        with open(self.model_path + 'train_logs/{}_log.txt'.format(self.model_name), 'w+') as log:
            print('\nUsing best model: Model {}\n'.format(best_model), file=log)
            print('Best Model Results:', file=log)
            print('Training Avg Diff: {:.3f}'.format(models_train_means[best_model]), file=log)
            print('Training Avg Diff Uncertainty: {:.3f}'.format(models_train_stds[best_model]), file=log)
            print('Testing Avg Diff: {:.3f}'.format(models_test_means[best_model]), file=log)
            print('Testing Avg Diff Uncertainty: {:.3f}'.format(models_test_stds[best_model]), file=log)
            print('Test Loss: {:.3f}'.format(models_test_final_loss[best_model]), file=log)
            print('Total Training Time: {:.2f} hrs'.format(train_time), file=log)
            print('\n')
            if self.debug:
                print('\nmodel saved at this point in no debug', file=log)
                return
        self.model = models[best_model]
        np.savez(self.model_path + 'train_logs/{}_train_history'.format(self.model_name),
                loss=models_train_lpe, val_loss=models_test_lpe, best_model=best_model, train_time=train_time)
        call(['rm','-r',self.npz_path + 'npz/'])
        return
    
    def load_Model(self, model_file):
        return
    
    def save_Model(self):
        self.model.save(self.model_path + self.model_name + '.h5')
        return

    def __rossNetAE(self, compression_size):
        '''
        Notes
        ------------
        Main architecture idea:
        Ref: https://doi.org/10.1029/2017JB015251
        '''
        input_seis = Input(shape=(self.total_time, 1))

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
        
        up1 = UpSampling1D(size=2)(reshaped)
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
                          activation='linear', padding='same')(conv_up3)
        
        model = Model(input_seis, decoding)
        
        model.compile(loss='mean_absolute_error',
                  optimizer=Adam(1e-4))
        
        return model
    
model = SortingModel('class')