#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 15:35:41 2018

@author: jorgeagr
"""
import os
import numpy as np
from tensorflow.keras.losses import Huber
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint
import obspy
from src.aux_funcs import check_String, read_Config

class PickingModel(object):
    
    def __init__(self, model_name):
        config = read_Config('models/{}.conf'.format(model_name))
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
        
        self.total_time = (self.window_before + self.window_after) * self.sample_rate
        
        if self.model_name not in os.listdir('models/'):
            for directory in [self.model_path, self.model_path+'train_logs/', self.model_path+'npz/']:
                #list(map(lambda x: x.format(self.model_name), ['models/{}/','models/{}/train_logs/', 'models/{}/npz/'])):
                os.mkdir(directory)
        return
    
    def __create_Train_Data(self):
        '''
        Function that iterates through seismograms in directory, perform preprocessing,
        create a time window around the arrival time and randomly shift it to augment
        the data set. Save augmented data set as npy files for quicker loading in
        the future. Meant for training/testing data.
        '''
        files = np.sort(os.listdir(self.files_path))
        gen_whitespace = lambda x: ' '*len(x)
        
        for f, file in enumerate(files):
            file = check_String(file)
            print_string = 'File ' + str(f+1) + ' / ' + str(len(files)) + '...'
            print('\r'+print_string, end=gen_whitespace(print_string))
            
            seismogram = obspy.read(self.files_path + file)
            seismogram = seismogram[0].resample(self.sample_rate)
            # Begging time
            b = seismogram.stats.sac['b']
            # The beginning time may not be 0, so shift all attribtues to be so
            shift = -b
            b = b + shift
            # End time
            e = seismogram.stats.sac['e'] + shift
            # Theoretical onset arrival time + shift
            th_arrival = seismogram.stats.sac[self.th_arrival_var] + shift
            # Picked maximum arrival time + shift
            arrival = seismogram.stats.sac[self.arrival_var] + shift
            
            # Theoretical arrival may be something unruly, so assign some random
            # shift from the picked arrival
            if not (b < th_arrival < e):
                th_arrival = arrival - 15 * np.random.rand()
            
            amp = seismogram.data
            time = seismogram.times()
            # Shifts + 1 because we want a 0th shift + 5 random ones
            rand_window_shifts = 2*np.random.rand(self.number_shift+1) - 1 # [-1, 1] interval
            abs_sort = np.argsort(np.abs(rand_window_shifts))
            rand_window_shifts = rand_window_shifts[abs_sort]
            rand_window_shifts[0] = 0
            
            seis_windows = np.zeros((self.number_shift+1, self.total_time, 1))
            arrivals = np.zeros((self.number_shift+1, 1))
            cut_time = np.zeros((self.number_shift+1, 1))
            for i, n in enumerate(rand_window_shifts):
                rand_arrival = th_arrival - n * self.window_shift
                init = np.where(np.round(rand_arrival - self.window_before, 1) == time)[0][0]
                end = np.where(np.round(rand_arrival + self.window_after, 1) == time)[0][0]
                time_i = time[init]
                time_f = time[end]
                if not (time_i < arrival < time_f):
                    time_i = arrival - 15 * np.random.rand() - self.window_before
                    time_f = time_i + self.window_after
                amp_i = amp[init:end]
                # Normalize by absolute peak, [-1, 1]
                amp_i = amp_i / np.abs(amp_i).max()
                seis_windows[i] = amp_i.reshape(self.total_time, 1)
                arrivals[i] = arrival - time[init]
                cut_time[i] = time[init]
            
            np.savez(self.model_path+'npz/{}'.format(file),
                     seis=seis_windows, arrival=arrivals, cut=cut_time)
            
        return
    
    def __train_test_split(self, idnum, seed=None):
        npz_files = np.sort(os.listdir('models/{}/npz'.format(self.model_name)))
        cutoff = int(len(npz_files) * (1-self.test_split))
        np.random.seed(seed)
        np.random.shuffle(npz_files)
        train_npz_list = npz_files[:cutoff]
        test_npz_list = npz_files[cutoff:]
        np.savez(self.model_path+'train_logs/train_test_split{}'.format(idnum),
                 train=train_npz_list, test=test_npz_list)
        
        return train_npz_list, test_npz_list
    
    def __load_Data(self, npz_list):
        seis_array = np.zeros((len(npz_list)*(self.number_shift+1), self.total_time, 1))
        arr_array = np.zeros((len(npz_list)*(self.number_shift+1), 1))
        for i, file in enumerate(npz_list):
            npz = np.load(self.model_path+'npz/'+file)
            for j in range(len(npz['seis'])):
                seis_array[i+self.number_shift*i+j] = npz['seis'][j]
                seis_array[i+self.number_shift*i+j] = npz['arrival'][j]
        return seis_array, arr_array
    
    def __get_Callbacks(self, epochs):
        stopper = EarlyStopping(monitor='val_loss', min_delta=0.001, 
                                patience=epochs//2, restore_best_weights=True)
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
    
        for m in range(self.model_iters):        
            print('Training arrival prediction model', m+1)
            model = self.__rossNet()
            
            callbacks = self.__get_Callbacks(self.epochs)
            
            train_files, test_files = self.__train_test_split(m)
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
        best_model = np.argmin(models_train_means)
        print('\nUsing best model: Model {}\n'.format(best_model + 1))
        print('Best Model Results:')
        print('Training Avg Diff: {:.3f}'.format(models_train_means[best_model]))
        print('Training Avg Diff Uncertainty: {:.3f}'.format(models_train_stds[best_model]))
        print('Testing Avg Diff: {:.3f}'.format(models_test_means[best_model]))
        print('Testing Avg Diff Uncertainty: {:.3f}'.format(models_test_stds[best_model]))
        print('Test Loss: {:.3f}'.format(models_test_final_loss[best_model]))
        print('\n')
        if self.debug:
            print('model saved at this point in no debug')
            return
        self.model = models[best_model]
        np.savez(self.model_path + 'train_logs/{}_train_history'.format(self.model_name),
                loss=models_train_lpe, val_loss=models_test_lpe, best_model=best_model)
        return
    
    def load_Model(self, model_file):
        return
    
    def save_Model(self):
        self.model.save(self.model_path + self.model_name + '.h5')
        return

    def __rossNet(self):
        '''
        Notes
        ------------
        Ref: https://doi.org/10.1029/2017JB015251 
        '''
        model = Sequential()
        model.add(Conv1D(32, 21, activation='relu',))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=2))
        
        model.add(Conv1D(64, 15, activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=2))
        
        model.add(Conv1D(128, 11, activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=2))
        
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(512, activation='relu'))
        model.add(Dense(1, activation='linear'))
        
        model.compile(loss=Huber(),
                      optimizer=Adam())
        
        return model