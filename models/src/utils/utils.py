# import modules
import os
import math
import numpy as np
import pandas as pd
import logging
from collections import OrderedDict

import torch
from torch import nn
from torch.optim import Optimizer
import torch.nn.functional as F


class activation():
    def __init__(self, act_type='leaky', negative_slope=0.2, inplace=True):
        super().__init__()
        self._act_type = act_type
        self.negative_slope = negative_slope
        self.inplace = inplace

    def __call__(self, input_):
        if self._act_type == 'leaky':
            return F.leaky_relu(input_, negative_slope=self.negative_slope, inplace=self.inplace)
        elif self._act_type == 'relu':
            return F.relu(input_, inplace=self.inplace)
        elif self._act_type == 'sigmoid':
            return torch.sigmoid(input_)
        else:
            raise NotImplementedError


class Adam16(Optimizer):
    """
    This version of Adam keeps an fp32 copy of the paramargseters and 
    does all of the parameter updates in fp32, while still doing the
    forwards and backwards passes using fp16 (i.e. fp16 copies of the 
    parameters and fp16 activations).
    Note that this calls .float().cuda() on the params such that it 
    moves them to gpu 0--if you're using a different GPU or want to 
    do multi-GPU you may need to deal with this.
    """
    
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, device=None):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay)
        params = list(params)
        super(Adam16, self).__init__(params, defaults)

        self.fp32_param_groups = [p.data.to(device=device, dtype=torch.float32) for p in params]
        if not isinstance(self.fp32_param_groups[0], dict):
            self.fp32_param_groups = [{'params': self.fp32_param_groups}]

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group, fp32_group in zip(self.param_groups, self.fp32_param_groups):
            for p, fp32_p in zip(group['params'], fp32_group['params']):
                if p.grad is None:
                    continue

                grad = p.grad.data.float()
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = grad.new().resize_as_(grad).zero_()
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = grad.new().resize_as_(grad).zero_()

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], fp32_p)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1
            
                # print(type(fp32_p))
                fp32_p.addcdiv_(-step_size, exp_avg, denom)
                p.data = fp32_p.half()

        return loss

def createfolder(directory):
    '''
    This function is used to create new folder with given directory.
    '''
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' +  directory)

def make_path(path, workfolder=None):
    '''
    This function is used to transform path to absolute path.
    '''
    if path[0] == '~':
        new_path = os.path.expanduser(path)
    else:
        new_path = path

    if workfolder is not None and not os.path.isabs(new_path):
        return os.path.join(workfolder, new_path)
    
    return new_path

def remove_file(file):
    if os.path.exists(file):
        os.remove(file)
        
def print_dict(d):
    for key, value in d.items():
        print('{}: {}'.format(key, value))


def save_model(epoch, optimizer, model, args):
    params_pt = os.path.join(args.params_folder, 'params_{}.pt'.format(epoch+1))
    remove_file(params_pt)
    # save the params per 10 epochs.
    torch.save({'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()},
                params_pt
                )

def get_logger(filename):
    # create logger
    remove_file(filename)
    logger = logging.getLogger('')
    logger.setLevel(logging.DEBUG)

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # create file handler and set level to debug
    f = logging.FileHandler(filename=filename)
    f.setLevel(logging.DEBUG)

    # create formatter
    formatter = logging.Formatter('%(message)s')

    # add formatter
    ch.setFormatter(formatter)
    f.setFormatter(formatter)

    # add handlers to logger
    logger.addHandler(ch)
    logger.addHandler(f)

    return logger

def make_layers(block):
    layers = []
    for layer_name, v in block.items():
        if 'pool' in layer_name:
            layer = nn.MaxPool2d(kernel_size=v[0], stride=v[1], padding=v[2])
            layers.append((layer_name, layer))
        elif 'deconv' in layer_name:
            transposeConv2d = nn.ConvTranspose2d(in_channels=v[0], out_channels=v[1],
                                                 kernel_size=v[2], stride=v[3],
                                                 padding=v[4])
            layers.append((layer_name, transposeConv2d))
            if 'relu' in layer_name:
                layers.append(('relu_' + layer_name, nn.ReLU(inplace=True)))
            elif 'leaky' in layer_name:
                layers.append(('leaky_' + layer_name, nn.LeakyReLU(negative_slope=0.2, inplace=True)))
        elif 'conv' in layer_name:
            conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1],
                               kernel_size=v[2], stride=v[3],
                               padding=v[4])
            layers.append((layer_name, conv2d))
            if 'relu' in layer_name:
                layers.append(('relu_' + layer_name, nn.ReLU(inplace=True)))
            elif 'leaky' in layer_name:
                layers.append(('leaky_' + layer_name, nn.LeakyReLU(negative_slope=0.2, inplace=True)))
        else:
            raise NotImplementedError

    return nn.Sequential(OrderedDict(layers))


def pixel_to_dBZ(img):
    """
    Parameters
    ----------
    img : np.ndarray or float
    Returns
    -------
    """
    return img * 90.0 - 10.0

def dBZ_to_pixel(dBZ_img):
    """
    Parameters
    ----------
    dBZ_img : np.ndarray
    Returns
    -------
    """
    return np.clip((dBZ_img + 10.0) / 90.0, a_min=0.0, a_max=1.0)


def pixel_to_rainfall(img, a=58.53, b=1.56):
    """Convert the pixel values to real rainfall intensity
    Parameters
    ----------
    img : np.ndarray
    a : float32, optional
    b : float32, optional
    Returns
    -------
    rainfall_intensity : np.ndarray
    """
    dBZ = pixel_to_dBZ(img)
    dBR = (dBZ - 10.0 * np.log10(a)) / b
    rainfall_intensity = np.power(10, dBR / 10.0)
    return rainfall_intensity


def rainfall_to_pixel(rainfall_intensity, a=58.53, b=1.56):
    """Convert the rainfall intensity to pixel values
    Parameters
    ----------
    rainfall_intensity : np.ndarray
    a : float32, optional
    b : float32, optional
    Returns
    -------
    pixel_vals : np.ndarray
    """
    dBR = np.log10(rainfall_intensity) * 10.0
    # dBZ = 10b log(R) +10log(a)
    dBZ = dBR * b + 10.0 * np.log10(a)
    pixel_vals = (dBZ + 10.0) / 90.0
    return pixel_vals

def dBZ_to_rainfall(dBZ, a=58.53, b=1.56):
    return np.power(10, (dBZ - 10 * np.log10(a))/(10*b))

def rainfall_to_dBZ(rainfall, a=58.53, b=1.56):
    return 10*np.log10(a) + 10*b*np.log10(rainfall)
