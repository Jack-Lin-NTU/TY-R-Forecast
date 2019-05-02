import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

import sys
sys.path.append("..")
from tools.args_tools import args

class ConvGRUCell(nn.Module):
    """
    Generate a convolutional GRU cell
    """
    def __init__(self, input_size, hidden_size, kernel_size, stride=1, padding=1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.reset_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, stride=stride, padding=padding)
        self.update_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, stride=stride, padding=padding)
        self.out_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, stride=stride, padding=padding)

        init.orthogonal_(self.reset_gate.weight)
        init.orthogonal_(self.update_gate.weight)
        init.orthogonal_(self.out_gate.weight)
        init.constant_(self.reset_gate.bias, 0.)
        init.constant_(self.update_gate.bias, 0.)
        init.constant_(self.out_gate.bias, 0.)


    def forward(self, input_, prev_state=None):

        # get batch and spatial sizes
        batch_size = input_.data.shape[0]
        spatial_size = input_.data.shape[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            if torch.cuda.is_available():
                prev_state = torch.zeros(state_size).to(args.device, dtype=torch.float)
            else:
                prev_state = torch.zeros(state_size)

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat([input_, prev_state], dim=1)
        update = torch.sigmoid(self.update_gate(stacked_inputs))
        reset = torch.sigmoid(self.reset_gate(stacked_inputs))
        out_inputs = F.leaky_relu(self.out_gate(torch.cat([input_, prev_state*reset], dim=1)), negative_slope=0.2)
        new_state = prev_state*update + out_inputs*(1-update)

        return new_state

class DeconvGRUCell(nn.Module):
    """
    Generate a deconvolutional GRU cell
    """
    def __init__(self, input_size, hidden_size, kernel_size, stride=1, padding=1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.reset_gate = nn.ConvTranspose2d(input_size+hidden_size, hidden_size, kernel_size, stride=stride, padding=padding)
        self.update_gate = nn.ConvTranspose2d(input_size+hidden_size, hidden_size, kernel_size, stride=stride, padding=padding)
        self.out_gate = nn.ConvTranspose2d(input_size+hidden_size, hidden_size, kernel_size, stride=stride, padding=padding)

        init.orthogonal_(self.reset_gate.weight)
        init.orthogonal_(self.update_gate.weight)
        init.orthogonal_(self.out_gate.weight)
        init.constant_(self.reset_gate.bias, 0.)
        init.constant_(self.update_gate.bias, 0.)
        init.constant_(self.out_gate.bias, 0.)


    def forward(self, input_=None, prev_state=None):
        # get batch and spatial sizes
        batch_size = prev_state.data.shape[0]
        spatial_size = prev_state.data.shape[2:]

        # data size is [batch, channel, height, width]
        if input_ is None:
            stacked_inputs = prev_state
        else:
            stacked_inputs = torch.cat([input_, prev_state], dim=1)
        update = torch.sigmoid(self.update_gate(stacked_inputs))
        reset = torch.sigmoid(self.reset_gate(stacked_inputs))

        if input_ is None:
            out_inputs = F.leaky_relu(self.out_gate(prev_state*reset),negative_slope=0.2)
        else:
            out_inputs = F.leaky_relu(self.out_gate(torch.cat([input_, prev_state*reset], dim=1)), negative_slope=0.2)

        new_state = prev_state*(1-update) + out_inputs*update

        return new_state