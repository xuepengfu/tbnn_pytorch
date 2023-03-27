# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 01:53:41 2022

@author: isfux
"""


import torch
import os
import numpy as np
import scipy.io as scio
from torch.utils.data import Dataset, DataLoader

class TensorTBNN(Dataset):
    '''
    x: input, five invariances. dim:N,5
    x1:tensor basis dim:N,10,9
    y: output, flatened reynold stress. dim:N,9
    '''
    def __init__(self, x, x1, y):
        self.x = torch.FloatTensor(x)
        self.x1 = torch.FloatTensor(x1)
        self.y = torch.FloatTensor(y)
        

    def __getitem__(self, idx):

        return self.x[idx], self.x1[idx], self.y[idx]

    def __len__(self):
        return len(self.x)




def main():
    dir = 'traindata_all.mat'
    data = scio.loadmat(dir);
    x = data['anisotropyRS_all']; 
    y = data['invariants_5'];  
    y1 = data['Tensorbasis'];  
    data = TensorTBNN(y, y1, x)
    loader = DataLoader(data, batch_size=32, shuffle=True)

    for x,y1,y in loader:
        print('sample:', x.shape, y1.shape, y.shape)


if __name__ == '__main__':
    main()

