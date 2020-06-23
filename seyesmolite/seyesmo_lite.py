#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 17:26:22 2020

@author: jorgeagr
"""

import os
import argparse
from src.models import PickingModel, SortingModel
from src.aux_funcs import check_String, read_Config

parser = argparse.ArgumentParser(description='Train a picking CNN using the RossNet architecture.')
parser.add_argument('model_name', help='Name of the model (barring the .conf extension)', type=str)
args = parser.parse_args()

model_name = args.model_name

config = read_Config('models/{}.conf'.format(model_name))
if config['model_type'] == 'p':
    model = PickingModel(model_name)
else:
    model = SortingModel(model_name)
model.train_Model()
model.save_Model()
