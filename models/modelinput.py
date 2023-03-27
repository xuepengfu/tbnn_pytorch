# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 03:03:38 2022

@author: isfux
"""

from models.MLP1 import MLP1
from models.MLP2 import MLP2
from models.MLP3 import MLP3
from models.MLP4 import MLP4

def select_model(args):
    type2model = {
        'MLP1': MLP1(),
        'MLP2': MLP2(),
        'MLP3': MLP3(),
        'MLP4': MLP4()
    }
    model = type2model[args.model_type]
    return model