# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 01:53:18 2022

@author: isfux
"""

from data.TensorTBNNDataset import TensorTBNN
import scipy.io as scio
import torch 
from torch.utils.data import random_split


def get_data(args):
    tempdata = scio.loadmat(args.datapath)
    x, x1, y = tempdata['invariants_5'], tempdata['Tensorbasis'], tempdata['anisotropyRSFU_all']
    return x, x1, y


def train_valid_split(inputs, tb, outputs, seed, valid_ratio = 0.1, test_ratio = 0):
    """
    Split provided training data into training, validation and test set
    :param inputs: scalar invariants
    :param tb: tensor basis
    :param outputs: reynold stress output   
    :valid_ratio: ratio of validation set between 0,1  always 0.1
    :test_ratio: ratio of test set between 0,1  always 0
    :seed:for fixing
    :return: training, validation and test set
    """
    num_points = inputs.shape[0]
    valid_set_size = int(valid_ratio * num_points) 
    test_set_size = int(test_ratio * num_points)   
    train_set_size = num_points - valid_set_size - test_set_size
    
    idx = range(num_points)
    train_set_inx, valid_set_inx, test_set_inx =\
    random_split(idx, [train_set_size, valid_set_size, test_set_size], generator=torch.Generator().manual_seed(seed))
    
    return inputs[train_set_inx, :], tb[train_set_inx, :], outputs[train_set_inx, :], \
           inputs[valid_set_inx, :], tb[valid_set_inx, :], outputs[valid_set_inx, :], \
           inputs[test_set_inx, :], tb[test_set_inx, :], outputs[test_set_inx, :]    


def get_dataset_by_type(args, x, x1, y):   
    type2data = {
        'TensorTBNNdata': TensorTBNN(x, x1, y)
    }
    dataset = type2data[args.data_type]
    return dataset    













