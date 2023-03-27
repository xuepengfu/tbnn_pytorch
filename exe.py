# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 20:46:42 2022

@author: isfux
"""
import os

inst = 'python train.py' + \
       ' --model_type MLP4' + \
       ' --seed 1337' + \
       ' --batch_size 256' + \
       ' --epochs 2' + \
       ' --lr 1e-5' + \
       ' --expnum 4'    
os.system(inst)



