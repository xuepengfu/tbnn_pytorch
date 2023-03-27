# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 19:47:38 2022

@author: isfux
"""
import time
old_time = time.time()

import math #常用数学常量及公式
import numpy as np #矩阵计算包

# Reading/Writing Data
import pandas as pd #like Excel
import os #输入输出
import csv #处理csv
import scipy.io as scio


# Pytorch
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader

#self-define 
from data.datainput import get_data, train_valid_split, get_dataset_by_type

from models.modelinput import select_model

from options import prepare_train_args

def same_seed(seed): 
    """
    Fixes random number generator seeds for reproducibility.
    :param seed: choose a lucky number
    :return:
    """
    torch.backends.cudnn.deterministic = True #固定框架的种子
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

        
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv1d') != -1:
        nn.init.kaiming_uniform_(m.weight.data)
        nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        nn.init.kaiming_uniform_(m.weight.data)
        nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

 
def trainer(train_loader, valid_loader, model, config, device):

    criterion = nn.MSELoss(reduction='mean') # Define your loss function, do not modify this.
   
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])    

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config['lr_decay'])


    n_epochs, best_loss, step, early_stop_count = config['n_epochs'], math.inf, 0, 0 #no regularization
    
    loss_record_plot = {'train': [], 'dev': []}      # for recording training loss
    
    lr_record = []

    model.apply(weights_init)
    for epoch in range(n_epochs):
        model.train() # Set your model to train mode.
        loss_record = []


        for x, x1, y in train_loader:
            optimizer.zero_grad()               # Set gradient to zero.
            x, x1, y = x.to(device), x1.to(device), y.to(device)   # Move your data to device. 
            pred = model(x,x1)                                                                
            loss = criterion(pred, y)
            loss.backward()                     # Compute gradient(backpropagation).
            optimizer.step()                    # Update parameters.
            step += 1
            loss_record.append(loss.detach().item())         
            

        mean_train_loss = sum(loss_record)/len(loss_record)
        loss_record_plot['train'].append(mean_train_loss)

        model.eval() # Set your model to evaluation mode.
        loss_record = []
        for x, x1, y in valid_loader:
            x, x1, y = x.to(device), x1.to(device), y.to(device)   # Move your data to device. 
            with torch.no_grad():
                pred = model(x,x1)            
                loss = model.criterion(pred, y)
            
            loss_record.append(loss.item())
            
        scheduler.step()
        
        lr_record.append(scheduler.get_last_lr()[0])
            
        mean_valid_loss = sum(loss_record)/len(loss_record)
        print(f'Epoch [{epoch+1}/{n_epochs}]: Train loss: {mean_train_loss:.4f}, Valid loss: {mean_valid_loss:.4f}')
        loss_record_plot['dev'].append(mean_valid_loss)

        if mean_valid_loss < best_loss:
            best_loss = mean_valid_loss
            torch.save(model.state_dict(), config['save_path']) # Save your best model
            print('Saving model with loss {:.3f}...'.format(best_loss))
            early_stop_count = 0
        else: 
            early_stop_count += 1

        if early_stop_count >= config['early_stop']:
            print('\nModel is not improving, so we halt the training session.')
            return loss_record_plot,lr_record
         
    return loss_record_plot,lr_record


args_train = prepare_train_args()


device = 'cuda' if torch.cuda.is_available() else 'cpu'
config = {
    'seed': args_train.seed,      # seed number
    'valid_ratio': args_train.validation_ratio, 
    'test_ratio': args_train.test_ratio,   
    'n_epochs': args_train.epochs,     # Number of epochs.            
    'batch_size': args_train.batch_size, 
    'learning_rate': args_train.lr,              
    'early_stop': args_train.earlystopnum,    # If model has not improved for this many consecutive epochs, stop training.  
    'save_path': args_train.modelsave_dir,  # Your model will be saved here.
    'lrsave_path': args_train.lrsave_dir,  
    'losssave_path': args_train.losssave_dir,     
    'lr_decay': args_train.lrdecay, 
    'weight_decay': args_train.weight_decay, 
    } 



same_seed(config['seed'])

invariants_data, tensorbasis_data, anisotropyRS_data = get_data(args_train)

inv_norm2 = np.linalg.norm(invariants_data, axis=0)

invariants_data = invariants_data/inv_norm2

invariants_data_mean = np.zeros((1, 5))
invariants_data_std = np.zeros((1, 5))
invariants_data_mean[0,:] = np.mean(invariants_data, axis=0)
invariants_data_std[0,:] = np.std(invariants_data, axis=0)

invariants_data = (invariants_data-invariants_data_mean)/invariants_data_std




invariants_train, tb_train, anirs_train, invariants_valid, tb_valid, anirs_valid,\
    invariants_test, tb_test, anirs_test = train_valid_split(invariants_data,\
         tensorbasis_data, anisotropyRS_data, config['seed'], config['valid_ratio'], config['test_ratio'])


# Print out the data size.
print(f"""invariants train_data size: {invariants_train.shape}
      anisotropy RS train_data size: {anirs_train.shape} 
      invariants valid_data size: {invariants_valid.shape}       
      anisotropy RS valid_data size: {anirs_valid.shape}           
     invariants test_data size: {invariants_test.shape}
     anisotropy RS  test_data size: {anirs_test.shape}""")


train_dataset, valid_dataset, test_dataset = get_dataset_by_type(args_train, invariants_train, tb_train, anirs_train), \
                                            get_dataset_by_type(args_train, invariants_valid, tb_valid, anirs_valid), \
                                             get_dataset_by_type(args_train, invariants_test, tb_test, anirs_test)



# Pytorch data loader loads pytorch dataset into batches.
train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True, num_workers=0, drop_last=True)
valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=True)


model = select_model(args_train).to(device)

print(model)
  

# In[8]
model_loss_record_plot, lr_record = trainer(train_loader, valid_loader, model, config, device)

scio.savemat(config['losssave_path'], model_loss_record_plot)
scio.savemat(config['lrsave_path'],  {'lr_record': lr_record})

current_time = time.time()

print("running time is " + str(current_time - old_time) + "s")
























           
           
           
           
           
           
           
           
           
           
           
           
           
           
           
           
           
           
           
           
           
           
           
           
           
           
           
           