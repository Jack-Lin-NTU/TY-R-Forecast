import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.tools.utils import make_layers, activation
from collections import OrderedDict

## CONVGRU cells
class ConvGRUcell(nn.Module):
    def __init__(self, c_input, c_hidden):
        super(ConvGRUcell, self).__init__()
        self.c_input = c_input
        self.c_hidden = c_hidden
        self.conv_gates = nn.Conv2d(c_input+c_hidden, c_hidden*2, kernel_size=3, stride=1, padding=1)
        self.out_gate = nn.Conv2d(c_input+c_hidden, c_hidden, kernel_size=3, stride=1, padding=1)
        self.dropout = nn.Dropout2d(0.1)
    
    def forward(self, inputs=None, states=None, seq_len=6):
        '''
        inputs size: B*S*C*H*W
        states size: B*C*H*W
        '''
        device = self.conv_gates.weight.device
        dtype = self.conv_gates.weight.dtype

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


## TRAJGRU cells
def warp(inputs, flow):
    B, C, H, W = inputs.shape
    yy, xx = torch.meshgrid(torch.arange(0, W), torch.arange(0, H))
    yy, xx = yy.cuda(), xx.cuda()
    xx = xx.view(1, 1, H, W).expand(B, -1, -1, -1)
    yy = yy.view(1, 1, H, W).expand(B, -1, -1, -1)
    grid = torch.cat((xx, yy), dim=1).float()
    vgrid = grid + flow

    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0
    vgrid = vgrid.permute(0, 2, 3, 1)
    output = F.grid_sample(inputs, vgrid)
    return output


class TrajGRUcell(nn.Module):
    def __init__(self, c_input, c_hidden, link_size, act_type, zoneout=0.0):
        super(TrajGRUcell, self).__init__()
        self.c_input = c_input
        self.c_hidden = c_hidden
        self.link_size = link_size
        self._zoneout = zoneout
        self._act_type = act_type

        self.flow_conv = nn.Sequential(
                                nn.Conv2d(c_input+c_hidden, 32, kernel_size=3, stride=1, padding=1),
                                nn.LeakyReLU(0.2),
                                nn.Conv2d(32, link_size*2, kernel_size=3, stride=1, padding=1)
        )
        self.i2h = nn.Conv2d(c_input, c_hidden*3, kernel_size=3, stride=1, padding=1)
        self.ret = nn.Conv2d(c_hidden*link_size, c_hidden*3, kernel_size=1, stride=1, padding=0)

    def _flow_generator(self, inputs=None, states=None):
        if inputs is None:
            i_shape = [states.shape[0], self.c_input] + list(states.shape[2:])
            inputs = torch.zeros(i_shape).to(device=self.device, dtype=self.dtype)
        flows = self.flow_conv(torch.cat([inputs, states], dim=1))
        flows = torch.split(flows, 2, dim=1)
        return flows
    
    def forward(self, inputs=None, states=None, seq_len=6):
        self.device = self.ret.weight.device
        self.dtype = self.ret.weight.dtype

        if states is None:
            h_shape = [inputs.shape[0], self.c_hidden] + list(inputs.shape[3:])
            states = torch.zeros(h_shape).to(device=self.device, dtype=self.dtype)
        if inputs is not None:
            B, S, C, H, W = inputs.shape
            i2h = self.i2h(torch.reshape(inputs, (-1, C, H, W)))
            i2h = torch.reshape(i2h, (B, S, i2h.shape[1], i2h.shape[2], i2h.shape[3]))
            i2h_slice = torch.split(i2h, self.c_hidden, dim=2)
        else:
            i2h_slice = None

        outputs = []
        
        prev_h = states
        outputs = []
        for i in range(seq_len):
            if inputs is not None:
                flows = self._flow_generator(inputs[:,i], prev_h)
            else:
                flows = self._flow_generator(None, prev_h)
            warpped_data = []
            for j in range(len(flows)):
                flow = flows[j]
                warpped_data.append(warp(prev_h, -flow))
            warpped_data = torch.cat(warpped_data, dim=1)
            # h2h size - 
            h2h = self.ret(warpped_data)
            h2h_slice = torch.split(h2h, self.c_hidden, dim=1)

            if i2h_slice is not None:
                reset_gate = torch.sigmoid(i2h_slice[0][:,i] + h2h_slice[0])
                update_gate = torch.sigmoid(i2h_slice[1][:,i]+ h2h_slice[1])
                new_mem = self._act_type(i2h_slice[2][:,i] + reset_gate * h2h_slice[2])
            else:
                reset_gate = torch.sigmoid(h2h_slice[0])
                update_gate = torch.sigmoid(h2h_slice[1])
                new_mem = self._act_type(reset_gate * h2h_slice[2])
            next_h = update_gate * prev_h + (1 - update_gate) * new_mem

            if self._zoneout > 0.0:
                mask = F.dropout2d(torch.zeros_like(prev_h), p=self._zoneout)
                next_h = torch.where(mask, next_h, prev_h)
            outputs.append(next_h)
            prev_h = next_h

        return torch.stack(outputs, dim=1), next_h


def get_cells(model):
    if 'CONVGRU' in model.upper():
        # build model
        encoder_elements = [
            [
                make_layers(OrderedDict({'conv1_leaky': [1, 8, 5, 3, 1]})),
                make_layers(OrderedDict({'conv2_leaky': [64, 192, 4, 2, 1]})),
                make_layers(OrderedDict({'conv3_leaky': [192, 192, 3, 2, 1]})),
            ],
            [
                ConvGRUcell(c_input=8, c_hidden=64),
                ConvGRUcell(c_input=192, c_hidden=192),
                ConvGRUcell(c_input=192, c_hidden=192),
            ]
        ]

        forecaster_elements = [
            [
                make_layers(OrderedDict({'deconv1_leaky': [192, 192, 3, 2, 1]})),
                make_layers(OrderedDict({'deconv2_leaky': [192, 64, 4, 2, 1]})),
                make_layers(OrderedDict({
                                        'deconv3_leaky': [64, 8, 5, 3, 1],
                                        'conv3_leaky': [8, 8, 3, 1, 1],
                                        'conv3': [8, 1, 1, 1, 0]
                                        })),
            ],
            [
                ConvGRUcell(c_input=192, c_hidden=192),
                ConvGRUcell(c_input=192, c_hidden=192),
                ConvGRUcell(c_input=64, c_hidden=64),
            ]
        ]

    elif 'TRAJGRU' in model.upper():
        # build model
        encoder_elements = [
            [
                make_layers(OrderedDict({'conv1_leaky': [1, 8, 5, 3, 1]})),
                make_layers(OrderedDict({'conv2_leaky': [64, 192, 4, 2, 1]})),
                make_layers(OrderedDict({'conv3_leaky': [192, 192, 3, 2, 1]})),
            ],
            [
                TrajGRUcell(c_input=8, c_hidden=64, link_size=13, act_type=activation()),
                TrajGRUcell(c_input=192, c_hidden=192, link_size=13, act_type=activation()),
                TrajGRUcell(c_input=192, c_hidden=192, link_size=9, act_type=activation()),
            ]
        ]

        forecaster_elements = [
            [
                make_layers(OrderedDict({'deconv1_leaky': [192, 192, 3, 2, 1]})),
                make_layers(OrderedDict({'deconv2_leaky': [192, 64, 4, 2, 1]})),
                make_layers(OrderedDict({
                                        'deconv3_leaky': [64, 8, 5, 3, 1],
                                        'conv3_leaky': [8, 8, 3, 1, 1],
                                        'conv3': [8, 1, 1, 1, 0]
                                        })),
            ],
            [
                TrajGRUcell(c_input=192, c_hidden=192, link_size=9, act_type=activation()),
                TrajGRUcell(c_input=192, c_hidden=192, link_size=13, act_type=activation()),
                TrajGRUcell(c_input=64, c_hidden=64, link_size=13, act_type=activation()),
            ]
        ]
    return encoder_elements, forecaster_elements