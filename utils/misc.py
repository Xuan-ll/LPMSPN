import os
import shutil
import torch 
import math
import numpy as np
import scipy.io
import matplotlib.pyplot as plt

def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array"
                         .format(type(tensor)))
    return tensor


def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray


def save_checkpoint(state, preds, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar', snapshot=None):
    preds = to_numpy(preds)
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    scipy.io.savemat(os.path.join(checkpoint, 'preds.mat'), mdict={'preds' : preds})

    if snapshot and state.epoch % snapshot == 0:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'checkpoint_{}.pth.tar'.format(state.epoch)))

    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))
        scipy.io.savemat(os.path.join(checkpoint, 'preds_best.mat'), mdict={'preds' : preds})

def copy_log(filepath = 'checkpoint'):
    filepath = os.path.join(filepath, 'log.txt')
    shutil.copyfile(filepath, os.path.join('log_backup.txt'))

def save_model(state, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filename = 'epoch'+str(state['epoch']) + filename
    filepath = os.path.join(checkpoint, filename)
    newpath = '/data1/ykx/xuan2/' + filename
#     torch.save(state, filepath)
    torch.save(state, newpath)
    # if snapshot and state.epoch % snapshot == 0:
    #     shutil.copyfile(filepath, os.path.join(checkpoint, 'checkpoint_{}.pth.tar'.format(state.epoch)))

def save_pred(preds, checkpoint='checkpoint', filename='preds_valid.mat'):
    preds = to_numpy(preds)
    filepath = os.path.join(checkpoint, filename)
    scipy.io.savemat(filepath, mdict={'preds' : preds})


def adjust_learning_rate(optimizer, epoch, schedule, gamma):
    """Sets the learning rate to the initial LR decayed by schedule"""
    if epoch in schedule:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= gamma
    return optimizer.state_dict()['param_groups'][0]['lr']

def adjust_lupi_wight(epoch):
    """Sets the learning rate to the initial LR decayed by schedule"""
#     if epoch <= 8:
#         return 0.02
#     elif epoch <= 16:
#         return 0.01
#     elif epoch <= 24:
#         return 0.005
#     else:
#         return 0.001
    if epoch <= 8:
        return 0.002
    elif epoch <= 16:
        return 0.001
    elif epoch <= 24:
        return 0.0005
    else:
        return 0.0001
#     if epoch <= 8:
#         return 0.004
#     elif epoch <= 16:
#         return 0.002
#     elif epoch <= 24:
#         return 0.001
#     else:
#         return 0.0001


