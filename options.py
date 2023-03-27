# -*- coding: utf-8 -*-


import argparse
import os


def parse_common_args(parser):
    """parse the parameters shared for training and testing, return the parser"""
    parser.add_argument('--model_type', type=str, default='MLP1', help='used in modelinput.py')
    parser.add_argument('--data_type', type=str, default='TensorTBNNdata', help='used in datainput.py')
    parser.add_argument('--datapath', type=str, default='traindata_onlySD_TBNN_5inv.mat', help='data dir')   
    parser.add_argument('--seed', type=int, default=1)
    return parser


def parse_train_args(parser):
    """parse the training-related parameters, return the parser"""
    parser = parse_common_args(parser)
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--lrdecay', type=float, default=0.9995, help='learning rate decay')
    parser.add_argument('--weight_decay', '--wd', default=0, type=float, help='weight decay')
    parser.add_argument('--earlystopnum', default=500, type=float, help='earlystop patience')   
    parser.add_argument('--expnum', type=str, default='1', help='number of exp')
    parser.add_argument('--modelsave_dir', type=str, default='./savemodels/model.ckpt', help='trained model save dir') 
    parser.add_argument('--lrsave_dir', type=str, default='./savemodels/loss_record.mat', help='lr save dir') 
    parser.add_argument('--losssave_dir', type=str, default='./savemodels/lr_record.mat', help='loss save dir') 
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--validation_ratio', type=float, default=0.1, help='validation set ratio')
    parser.add_argument('--test_ratio', type=float, default=0, help='test set ratio')  
    parser.add_argument('--model_dir', type=str, default='', help='leave blank, auto generated')
    return parser


def parse_test_args(parser):
    """parse the validating/testing-related parameters, return the parser"""
    parser = parse_common_args(parser)
    return parser


def get_common_args():
    """return the args(training)"""
    parser = argparse.ArgumentParser()
    parser = parse_common_args(parser)
    args = parser.parse_args()
    return args

def get_train_args():
    """return the args(training)"""
    parser = argparse.ArgumentParser()
    parser = parse_train_args(parser)
    args = parser.parse_args()
    return args


def get_test_args():
    """return the args(validating/testing)"""
    parser = argparse.ArgumentParser()
    parser = parse_test_args(parser)
    args = parser.parse_args()
    return args


def get_train_model_dir(args):
    model_dir = os.path.join('checkpoints', args.model_type + '_' + args.expnum)
    if not os.path.exists(model_dir):
        os.system('mkdir -p ' + model_dir)
    args.model_dir = model_dir


def save_args(args, save_dir):
    args_path = os.path.join(save_dir, 'args.txt')
    with open(args_path, 'w') as fd:
        fd.write(str(args).replace(', ', ',\n'))


def get_model_loss_lr_trainedmodel_dir(args):
    modelsave_dir = os.path.join(args.model_dir,'model.ckpt')
    lrsave_dir = os.path.join(args.model_dir,'lr_record.mat')
    losssave_dir = os.path.join(args.model_dir,'loss_record.mat')    
    args.modelsave_dir = modelsave_dir
    args.lrsave_dir = lrsave_dir
    args.losssave_dir = losssave_dir
      

def prepare_train_args():
    """create a parser that contains all the shared and training parameters. then create a model directory and call
    save_args to save all the parameters and save corresponding args """
    args = get_train_args()
    get_train_model_dir(args)
    get_model_loss_lr_trainedmodel_dir(args)
    save_args(args, args.model_dir)
    return args    

def prepare_train_args_fortest():
    """create a parser that contains all the shared and training parameters. then create a model directory and call
    save_args to save all the parameters and save corresponding args """
    args = get_train_args()
    get_model_loss_lr_trainedmodel_dir(args)
    return args  



if __name__ == '__main__':
    train_args = prepare_train_args()
    train_args.losssave_dir
    train_args.modelsave_dir
    train_args.model_dir
    train_args.datapath




