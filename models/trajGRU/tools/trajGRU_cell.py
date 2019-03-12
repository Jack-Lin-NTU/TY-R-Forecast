import sys
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from .args_tools import args

class subCNN(nn.Module):
    def __init__(self, channel_input, channel_hidden, link_size):
        super().__init__()
        self.channel_input = channel_input
        self.channel_hidden = channel_hidden
        self.link_size = link_size
        layer_sublist = []
        layer_sublist.append(nn.Conv2d(channel_input+channel_hidden, 32, 5, 1, 2))
        layer_sublist.append(nn.ReLU())
        layer_sublist.append(nn.Conv2d(32, link_size*2, 5, 1, 2))
        layer_sublist.append(nn.ReLU())

        nn.init.orthogonal_(layer_sublist[0].weight)
        nn.init.constant_(layer_sublist[0].bias, 0.)
        nn.init.orthogonal_(layer_sublist[2].weight)
        nn.init.constant_(layer_sublist[2].bias, 0.)

        self.layer = nn.Sequential(*layer_sublist)

    def grid_sample(self, input, grids_x, grids_y):
        b, c, h, w = input.shape
        l = grids_x.shape[1]
        input = input[:,:,None,:,:].expand((b,c,l,h,w))
        grids_x = grids_x.unsqueeze(4)
        grids_y = grids_y.unsqueeze(4)
        grids_l = torch.arange(1,l+1).unsqueeze(1).unsqueeze(2).unsqueeze(3).expand((b,l,h,w,1)).to(args.device, dtype=torch.float)
        grids = torch.cat([grids_x/(w-1), grids_y/(h-1), grids_l/(l-1)],4)
        grids = grids*2-1

        return F.grid_sample(input, grids)

    def forward(self, input=None, prev_state=None):
        # get batch and spatial sizes
        if self.channel_input == 0:
            stacked_inputs = prev_state
        else:
            stacked_inputs = torch.cat([input, prev_state], dim=1)

        output = self.layer(stacked_inputs)

        return self.grid_sample(prev_state, output[:,0:self.link_size,:,:], output[:,self.link_size:,:,:])


class warp_CNN(nn.Module):
    def __init__(self, channel_hidden, link_size, kernel_size=1, stride=1, padding=0):
        super().__init__()
        self.channel_hidden = channel_hidden
        self.link_size = link_size

        self.warpnet = nn.Conv2d(channel_hidden*link_size, channel_hidden, kernel_size, stride, padding)

        nn.init.orthogonal_(self.warpnet.weight)
        nn.init.constant_(self.warpnet.bias, 0.)

    def forward(self, M):
        # M shape = B x C x L x H x W
        B, C, L, H, W = M.shape
        M = M.view(B, C*L, H, W)

        return self.warpnet(M)

class warp_DECNN(nn.Module):
    def __init__(self, channel_hidden, link_size, kernel_size=1, stride=1, padding=0):
        super().__init__()
        self.channel_hidden = channel_hidden
        self.link_size = link_size

        self.warpnet = nn.ConvTranspose2d(channel_hidden*link_size, channel_hidden, kernel_size, stride, padding)

        nn.init.orthogonal_(self.warpnet.weight)
        nn.init.constant_(self.warpnet.bias, 0.)

    def forward(self, M):
        # M shape = B x C x L x H x W
        B, C, L, H, W = M.shape
        M = M.view(B, C*L, H, W)
        return self.warpnet(M)

class trajGRUCell(nn.Module):
    """
    Generate a convolutional traj GRU cell
    """
    def __init__(self, channel_input, channel_hidden, link_size, kernel_size, stride=1, padding=1, batch_norm=False):
        super().__init__()
        self.channel_input = channel_input
        self.channel_hidden = channel_hidden
        self.link_size = link_size

        self.subnetwork = subCNN(channel_input, channel_hidden, link_size)

        self.reset_gate_input = nn.Conv2d(channel_input, channel_hidden, kernel_size, stride, padding)
        self.update_gate_input = nn.Conv2d(channel_input, channel_hidden, kernel_size, stride, padding)
        self.out_gate_input = nn.Conv2d(channel_input, channel_hidden, kernel_size, stride, padding)

        self.reset_gate_warp = warp_CNN(channel_hidden, link_size, 1, 1, 0)
        self.update_gate_warp = warp_CNN(channel_hidden, link_size, 1, 1, 0)
        self.out_gate_warp = warp_CNN(channel_hidden, link_size, 1, 1, 0)

        init.orthogonal_(self.reset_gate_input.weight)
        init.orthogonal_(self.update_gate_input.weight)
        init.orthogonal_(self.out_gate_input.weight)
        init.constant_(self.reset_gate_input.bias, 0.)
        init.constant_(self.update_gate_input.bias, 0.)
        init.constant_(self.out_gate_input.bias, 0.)


    def forward(self, input, prev_state=None):
        # get batch and spatial sizes
        batch_size = input.data.shape[0]
        H, W = input.data.shape[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = (batch_size, self.channel_hidden, H, W)
            if torch.cuda.is_available():
                prev_state = torch.zeros(state_size).to(args.device, dtype=torch.float)
            else:
                prev_state = torch.zeros(state_size)

        # data size is [batch, channel, height, width]
        M = self.subnetwork(input, prev_state)
        update = torch.sigmoid(self.update_gate_input(input)+self.update_gate_warp(M))
        reset = torch.sigmoid(self.reset_gate_input(input)+self.reset_gate_warp(M))
        out_inputs = F.leaky_relu((self.out_gate_input(input)+reset*self.out_gate_warp(M)), negative_slope=0.2)
        new_state = prev_state*update + out_inputs*(1-update)
        return new_state

class DetrajGRUCell(nn.Module):
    """
    Generate a deconvolutional traj GRU cell
    """
    def __init__(self, channel_input, channel_hidden, link_size, kernel_size, stride=1, padding=1, batch_norm=False):
        super().__init__()
        self.channel_input = channel_input
        self.channel_hidden = channel_hidden

        self.subnetwork = subCNN(channel_input, channel_hidden, link_size)
        if channel_input != 0:
            self.reset_gate_input = nn.ConvTranspose2d(channel_input, channel_hidden, kernel_size, stride, padding)
            self.update_gate_input = nn.ConvTranspose2d(channel_input, channel_hidden, kernel_size, stride, padding)
            self.out_gate_input = nn.ConvTranspose2d(channel_input, channel_hidden, kernel_size, stride, padding)
            init.orthogonal_(self.reset_gate_input.weight)
            init.orthogonal_(self.update_gate_input.weight)
            init.orthogonal_(self.out_gate_input.weight)
            init.constant_(self.reset_gate_input.bias, 0.)
            init.constant_(self.update_gate_input.bias, 0.)
            init.constant_(self.out_gate_input.bias, 0.)

        self.reset_gate_warp = warp_DECNN(channel_hidden, link_size, 1, 1, 0)
        self.update_gate_warp = warp_DECNN(channel_hidden, link_size, 1, 1, 0)
        self.out_gate_warp = warp_DECNN(channel_hidden, link_size, 1, 1, 0)

    def forward(self, input=None, prev_state=None):
        # get batch and spatial sizes
        batch_size = prev_state.data.shape[0]
        H, W = prev_state.data.shape[2:]

        if self.channel_input == 0:
            M = self.subnetwork(prev_state=prev_state)
            update = torch.sigmoid(self.update_gate_warp(M))
            reset = torch.sigmoid(self.reset_gate_warp(M))
            out_inputs = F.leaky_relu(reset*self.out_gate_warp(M), negative_slope=0.2)
            new_state = prev_state*(1-update) + out_inputs*update

        else:
            M = self.subnetwork(input=input, prev_state=prev_state)
            update = torch.sigmoid(self.update_gate_input(input)+self.update_gate_warp(M))
            reset = torch.sigmoid(self.reset_gate_input(input)+self.reset_gate_warp(M))
            out_inputs = F.leaky_relu((self.out_gate_input(input)+reset*self.out_gate_warp(M)), negative_slope=0.2)
            new_state = prev_state*(1-update) + out_inputs*update

        return new_state