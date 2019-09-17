import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvGRUcell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ConvGRUcell. self).__init__()
        self.c_input = input_size
        self.c_hidden = hidden_size
        self.conv_gates = nn.Conv2d(input_size+hidden_size, hidden_size*2, kernel_size=3, stride=1, padding=1)
        self.out_gate = nn.Conv2d(input_size, hidden_size, kernel_size=3, stride=1, padding=1)
        self.dropout = nn.Dropout2d(0.1)
    
    def forward(self, input_=None, hidden=None):
        device = self.reset_gate.weight.device
        dtype = self.reset_gate.weight.dtype

        if hidden is None:
            h_shape = [input_.shape[0], self.c_hidden] + list(input_.shape[2:])
            hidden = torch.zeros(h_shape).to(device=device, dtype=dtype)

        if input_ is None:
            i_shape = [hidden.shape[0], self.c_hidden] + list(hidden.shape[2:])
            input_ = torch.zeros(i_shape).to(device=device, dtype=dtype)

        tmp = self.conv_gates(torch.cat([input_, hidden], dim=1))
        (rt, ut) = tmp.chunk(2,1)

        reset_gate = self.dropout(torch.sigmoid(rt))
        update_gate = self.dropout(torch.sigmoid(ut))
        condi_h = F.leaky_relu(self.out_gate(torch.cat([input_, reset_gate*hidden], dim=1))), negative_slope=0.2)
        new_h = update_gate*hidden + (1-update_gate)*new_hidden

        return new_h