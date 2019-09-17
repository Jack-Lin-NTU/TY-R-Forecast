import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

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


class TrajGRU(nn.Module):
    def __init__(self, input_size, hidden_size, link_size, act_type):
        super(TrajGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.link_size = link_size

        self.flow_conv = nn.Sequential(
                                nn.Conv2d(input_size+hidden_size, 32, kernel_size=3, stride=1, padding=1),
                                nn.LeakyReLU(0.2),
                                nn.Conv2d(32, link_size, kernel_size=3, stride=1, padding=1)
        )

        self.ret = nn.Conv2d(hidden_size*link_size, hidden_size, kernel_size=1, padding=1, padding=0)

    def _flow_generator(self, inputs=None, states=None):
        if inputs is None:
            i_shape = [states.shape[0], self.input_size] + list(states.shape[2:])
            inputs = torch.zeros(i_shape).to(device=self.device, dtype=self.dtype)
        flows = self.flow_conv(torch.cat([inputs, states],dim=1))
        return flows
    
    def forward(self, inputs=None, states=None, seq_len=6):
        self.device = self.ret.weight.device
        self.dtype = self.ret.weight.dtype

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
            