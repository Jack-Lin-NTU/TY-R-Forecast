import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvGRUcell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ConvGRUcell, self).__init__()
        self.c_input = input_size
        self.c_hidden = hidden_size
        self.conv_gates = nn.Conv2d(input_size+hidden_size, hidden_size*2, kernel_size=3, stride=1, padding=1)
        self.out_gate = nn.Conv2d(input_size+hidden_size, hidden_size, kernel_size=3, stride=1, padding=1)
        self.dropout = nn.Dropout2d(0.1)
    
    def forward(self, inputs=None, states=None):
        '''
        inputs size: B*S*C*H*W
        states size: B*C*H*W
        '''
        device = self.conv_gates.weight.device
        dtype = self.conv_gates.weight.dtype
        seq_len = 
        if states is None:
            h_shape = [inputs.shape[0], self.c_hidden] + list(inputs.shape[3:])
            states = torch.zeros(h_shape).to(device=device, dtype=dtype)
        
        outputs = []

        for i in range(seq_len):
            if inputs is None:
                i_shape = [states.shape[0], self.c_hidden] + list(states.shape[2:])
                x = torch.zeros(i_shape).to(device=device, dtype=dtype)
            else:
                x = inputs[:,i]

            tmp = self.conv_gates(torch.cat([x, states], dim=1))
            (rt, ut) = tmp.chunk(2,1)

            reset_gate = self.dropout(torch.sigmoid(rt))
            update_gate = self.dropout(torch.sigmoid(ut))
            condi_h = F.leaky_relu(self.out_gate(torch.cat([x, reset_gate*states], dim=1)), negative_slope=0.2)
            new_h = update_gate*states + (1-update_gate)*condi_h
            outputs.append(new_h)
        return torch.stack(outputs, dim=1), new_h