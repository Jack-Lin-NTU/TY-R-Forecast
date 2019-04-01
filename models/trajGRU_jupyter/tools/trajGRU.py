import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from .cnn2D_model import *
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
        grids_l = torch.arange(1,l+1).unsqueeze(1).unsqueeze(2).unsqueeze(3).expand((b,l,h,w,1)).to(args.device, dtype=args.value_dtype)
        grids = torch.cat([grids_x/(w-1), grids_y/(h-1), grids_l/(l-1)], 4)
        grids = grids*2-1

        return F.grid_sample(input, grids)

    def forward(self, input=None, prev_state=None):
        # get batch and spatial sizes
#         print('Prev:',prev_state.shape)
        if self.channel_input == 0:
            stacked_inputs = prev_state
        else:
#             print('Input:',input.shape)
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
                prev_state = torch.zeros(state_size).to(args.device, dtype=args.value_dtype)
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



class Encoder(nn.Module):
    def __init__(self, channel_input, channel_downsample, channel_rnn,
                downsample_k, downsample_s, downsample_p,
                rnn_link_size, rnn_k, rnn_s, rnn_p, n_layers,
                batch_norm=False):
        '''
        Generates a multi-layer convolutional GRU.
        Preserves spatial dimensions across cells, only altering depth.
        Parameters
        ----------
        channel_input: (integer.) depth dimension of input tensors.
        channel_downsample: (integer or list.) depth dimensions of downsample.
        channel_rnn: (integer or list.) depth dimensions of rnn.
        rnn_link_size: (integer or list.) link size to select the import points in hidden states in rnn.
        downsample_k: (integer or list.) the kernel size of each downsample layers.
        downsample_s: (integer or list.) the stride size of each downsample layers.
        downsample_p: (integer or list.) the padding size of each downsample layers.
        rnn_k: (integer or list.) the kernel size of each rnn layers.
        rnn_s: (integer or list.) the stride size of each rnn layers.
        rnn_p: (integer or list.) the padding size of each rnn layers.
        n_layers: (integer.) number of chained "ConvGRUCell".
        '''
        super().__init__()

        self.channel_input = channel_input

        # channel size
        if type(channel_downsample) != list:
            self.channel_downsample = [channel_downsample]*int(n_layers/2)
        else:
            assert len(channel_downsample) == int(n_layers/2), '`channel_downsample` must have the same length as n_layers/2'
            self.channel_downsample = channel_downsample

        if type(channel_rnn) != list:
            self.channel_rnn = [channel_rnn]*int(n_layers/2)
        else:
            assert len(channel_rnn) == int(n_layers/2), '`channel_rnn` must have the same length as n_layers/2'
            self.channel_rnn = channel_rnn

        if type(rnn_link_size) != list:
            self.rnn_link_size = [rnn_link_size]*int(n_layers/2)
        else:
            assert len(rnn_link_size) == int(n_layers/2), '`rnn_link_size` must have the same length as n_layers/2'
            self.rnn_link_size = rnn_link_size

        # kernel size
        if type(downsample_k) != list:
            self.downsample_k = [downsample_k]*int(n_layers/2)
        else:
            assert len(downsample_k) == int(n_layers/2), '`downsample_k` must have the same length as n_layers/2'
            self.downsample_k = downsample_k

        if type(rnn_k) != list:
            self.rnn_k = [rnn_k]*int(n_layers/2)
        else:
            assert len(rnn_k) == int(n_layers/2), '`rnn_k` must have the same length as n_layers/2'
            self.rnn_k = rnn_k

       # stride size
        if type(downsample_s) != list:
            self.downsample_s = [downsample_s]*int(n_layers/2)
        else:
            assert len(downsample_s) == int(n_layers/2), '`downsample_s` must have the same length as n_layers/2'
            self.downsample_s = downsample_s

        if type(rnn_s) != list:
            self.rnn_s = [rnn_s]*int(n_layers/2)
        else:
            assert len(rnn_s) == int(n_layers/2), '`rnn_s` must have the same length as n_layers/2'
            self.rnn_s = rnn_s

        # padding size
        if type(downsample_p) != list:
            self.downsample_p = [downsample_p]*int(n_layers/2)
        else:
            assert len(downsample_p) == int(n_layers/2), '`downsample_p` must have the same length as n_layers/2'
            self.downsample_p = downsample_p

        if type(rnn_p) != list:
            self.rnn_p = [rnn_p]*int(n_layers/2)
        else:
            assert len(rnn_p) == int(n_layers/2), '`rnn_p` must have the same length as n_layers/2'
            self.rnn_p = rnn_p

        self.n_layers = n_layers

        cells = []
        for i in range(int(self.n_layers/2)):
            if i == 0:
                cell=CNN2D_cell(self.channel_input, self.channel_downsample[i], self.downsample_k[i],
                                self.downsample_s[i], self.downsample_p[i], batch_norm=batch_norm)
            else:
                cell=CNN2D_cell(self.channel_rnn[i-1], self.channel_downsample[i], self.downsample_k[i],
                                self.downsample_s[i], self.downsample_p[i], batch_norm=batch_norm)

            name = 'Downsample_' + str(i).zfill(2)
            setattr(self, name, cell)
            cells.append(getattr(self, name))

            cell = trajGRUCell(self.channel_downsample[i], self.channel_rnn[i], self.rnn_link_size[i],
                                self.rnn_k[i], self.rnn_s[i], self.rnn_p[i])
            name = 'trajGRUCell_' + str(i).zfill(2)
            setattr(self, name, cell)
            cells.append(getattr(self, name))

        self.cells = cells


    def forward(self, x, hidden=None):
        if not hidden:
            hidden = [None]*int(self.n_layers/2)

        input_ = x
        upd_hidden = []

        for i in range(self.n_layers):
            if i % 2 == 0:
                cell = self.cells[i]
                input_ = cell(input_)
            else:
                cell = self.cells[i]
                cell_hidden = hidden[int(i/2)]

                # pass through layer
                upd_cell_hidden = cell(input_, cell_hidden)
                upd_hidden.append(upd_cell_hidden)
                # update input_ to the last updated hidden layer for next pass
                input_ = upd_cell_hidden
        # retain tensors in list to allow different hidden sizes
        return upd_hidden


class Forecaster(nn.Module):
    def __init__(self, channel_input, channel_upsample, channel_rnn,
                upsample_k, upsample_p, upsample_s,
                rnn_link_size, rnn_k, rnn_s, rnn_p, n_layers,
                channel_output=1, output_k=1, output_s = 1, output_p=0, n_output_layers=1,
                batch_norm=False):
        '''
        Generates a multi-layer convolutional GRU.
        Preserves spatial dimensions across cells, only altering depth.
        Parameters
        ----------
        channel_input: (integer.) depth dimension of input tensors.
        channel_upsample: (integer or list.) depth dimensions of upsample.
        channel_rnn: (integer or list.) depth dimensions of rnn.
        rnn_link_size: (integer or list.) link size to select the import points in hidden states in rnn.
        upsample_s: (integer or list.) the stride size of upsample layers.
        upsample_p: (integer or list.) the padding size of upsample layers.
        rnn_k: (integer or list.) the kernel size of rnn layers.
        rnn_s: (integer or list.) the stride size of rnn layers.
        rnn_p: (integer or list.) the padding size of rnn layers.
        n_layers: (integer.) number of chained "DeconvGRUCell".
        ## output layer params
        channel_output: (integer or list.) depth dimensions of output.
        output_k: (integer or list.) the kernel size of output layers.
        output_s: (integer or list.) the stride size of output layers.
        output_p: (integer or list.) the padding size of output layers.
        n_output_layers=1
        '''
        super().__init__()

        self.channel_input = channel_input
        # channel size
        if type(channel_upsample) != list:
            self.channel_upsample = [channel_upsample]*int(n_layers/2)
        else:
            assert len(channel_upsample) == int(n_layers/2), '`channel_upsample` must have the same length as n_layers/2'
            self.channel_upsample = channel_upsample

        if type(channel_rnn) != list:
            self.channel_rnn = [channel_rnn]*int(n_layers/2)
        else:
            assert len(channel_rnn) == int(n_layers/2), '`channel_rnn` must have the same length as n_layers/2'
            self.channel_rnn = channel_rnn

        if type(rnn_link_size) != list:
            self.rnn_link_size = [rnn_link_size]*int(n_layers/2)
        else:
            assert len(rnn_link_size) == int(n_layers/2), '`rnn_link_size` must have the same length as n_layers/2'
            self.rnn_link_size = rnn_link_size

        # kernel size
        if type(upsample_k) != list:
            self.upsample_k = [upsample_k]*int(n_layers/2)
        else:
            assert len(upsample_k) == int(n_layers/2), '`upsample_k` must have the same length as n_layers/2'
            self.upsample_k = upsample_k

        if type(rnn_k) != list:
            self.rnn_k = [rnn_k]*int(n_layers/2)
        else:
            assert len(rnn_k) == int(n_layers/2), '`rnn_k` must have the same length as n_layers/2'
            self.rnn_k = rnn_k

       # stride size
        if type(upsample_s) != list:
            self.upsample_s = [upsample_s]*int(n_layers/2)
        else:
            assert len(upsample_s) == int(n_layers/2), '`upsample_s` must have the same length as n_layers/2'
            self.upsample_s = upsample_s

        if type(rnn_s) != list:
            self.rnn_s = [rnn_s]*int(n_layers/2)
        else:
            assert len(rnn_s) == int(n_layers/2), '`rnn_s` must have the same length as n_layers/2'
            self.rnn_s = rnn_s

        # padding size
        if type(upsample_p) != list:
            self.upsample_p = [upsample_p]*int(n_layers/2)
        else:
            assert len(upsample_p) == int(n_layers/2), '`upsample_p` must have the same length as n_layers/2'
            self.upsample_p = upsample_p

        if type(rnn_p) != list:
            self.rnn_p = [rnn_p]*int(n_layers/2)
        else:
            assert len(rnn_p) == int(n_layers/2), '`rnn_p` must have the same length as n_layers/2'
            self.rnn_p = rnn_p

        # output size
        if type(channel_output) != list:
            self.channel_output = [channel_output]*int(n_output_layers)
        else:
            assert len(channel_output) == int(n_output_layers), '`channel_output` must have the same length as n_output_layers'
            self.channel_output = channel_output

        if type(output_k) != list:
            self.output_k = [output_k]*int(n_output_layers)
        else:
            assert len(output_k) == int(n_output_layers), '`output_k` must have the same length as n_output_layers'
            self.output_k = output_k

        if type(output_p) != list:
            self.output_p = [output_p]*int(n_output_layers)
        else:
            assert len(output_p) == int(n_output_layers), '`output_p` must have the same length as n_output_layers'
            self.output_p = output_p

        if type(output_s) != list:
            self.output_s = [output_s]*int(n_output_layers)
        else:
            assert len(output_s) == int(n_output_layers), '`output_s` must have the same length as n_output_layers'
            self.output_s = output_s

        self.n_output_layers = n_output_layers
        self.n_layers = n_layers

        cells = []
        for i in range(int(self.n_layers/2)):
            if i == 0:
                cell = DetrajGRUCell(self.channel_input, self.channel_rnn[i], self.rnn_link_size[i],
                                    self.rnn_k[i], self.rnn_s[i], self.rnn_p[i])
            else:
                cell = DetrajGRUCell(self.channel_upsample[i-1], self.channel_rnn[i], self.rnn_link_size[i],
                                    self.rnn_k[i], self.rnn_s[i], self.rnn_p[i])

            name = 'DetrajGRUCelll_' + str(i).zfill(2)
            setattr(self, name, cell)
            cells.append(getattr(self, name))

            cell = DeCNN2D_cell(self.channel_rnn[i], self.channel_upsample[i], self.upsample_k[i], self.upsample_s[i],
                                self.upsample_p[i], batch_norm=batch_norm)
            name = 'Upsample_' + str(i).zfill(2)
            setattr(self, name, cell)
            cells.append(getattr(self, name))


        for i in range(self.n_output_layers):
            if i == 0:
                cell = nn.Conv2d(self.channel_upsample[-1], self.channel_output[i], self.output_k[i], self.output_s[i], self.output_p[i])
            else:
                cell = nn.Conv2d(self.channel_output[i-1], self.channel_output[i], self.output_k[i], self.output_s[i], self.output_p[i])
        name = 'OutputLayer_' + str(i).zfill(2)
        setattr(self, name, cell)
        cells.append(getattr(self, name))
        self.cells = cells

    def forward(self, hidden):
        '''
        Parameters
        ----------
        x : 4D input tensor. (batch, channels, height, width).
        hidden : list of 4D hidden state representations. (batch, channels, height, width).
        Returns
        -------
        upd_hidden : 5D hidden representation. (layer, batch, channels, height, width).
        '''
        input_ = None

        upd_hidden = []
        output = 0

        for i in range(self.n_layers):
            if i % 2 == 0:
                cell = self.cells[i]
                cell_hidden = hidden[int(i/2)]
                # pass through layer

                upd_cell_hidden = cell(input=input_, prev_state=cell_hidden)

                upd_hidden.append(upd_cell_hidden)
                # update input_ to the last updated hidden layer for next pass
                input_ = upd_cell_hidden
            else:
                cell = self.cells[i]
                input_ = cell(input_)
        cell = self.cells[-1]
        output = cell(input_)

        # retain tensors in list to allow different hidden sizes
        return upd_hidden, output


class model(nn.Module):
    def __init__(self, n_encoders, n_forecasters, rnn_link_size,
                encoder_input_channel, encoder_downsample_channels, encoder_rnn_channels,
                encoder_downsample_k, encoder_downsample_s, encoder_downsample_p,
                encoder_rnn_k, encoder_rnn_s, encoder_rnn_p, encoder_n_layers,
                forecaster_input_channel, forecaster_upsample_channels, forecaster_rnn_channels,
                forecaster_upsample_k, forecaster_upsample_s, forecaster_upsample_p,
                forecaster_rnn_k, forecaster_rnn_s, forecaster_rnn_p, forecaster_n_layers,
                forecaster_output=1, forecaster_output_k=1, forecaster_output_s=1, forecaster_output_p=0, forecaster_output_layers=1,
                batch_norm=False):

        super().__init__()
        self.n_encoders = n_encoders
        self.n_forecasters = n_forecasters

        models = []
        for i in range(self.n_encoders):
            model = Encoder(channel_input=encoder_input_channel, channel_downsample=encoder_downsample_channels, channel_rnn=encoder_rnn_channels,
                            downsample_k=encoder_downsample_k, downsample_s=encoder_downsample_s, downsample_p=encoder_downsample_p,
                            rnn_link_size=rnn_link_size, rnn_k=encoder_rnn_k, rnn_s=encoder_rnn_s, rnn_p=encoder_rnn_p,
                            n_layers=encoder_n_layers, batch_norm=batch_norm)
            name = 'Encoder_' + str(i+1).zfill(2)
            setattr(self, name, model)
            models.append(getattr(self, name))

        for i in range(self.n_forecasters):
            model = Forecaster(channel_input=forecaster_input_channel, channel_upsample=forecaster_upsample_channels,
                                channel_rnn=forecaster_rnn_channels,
                                upsample_k=forecaster_upsample_k, upsample_s=forecaster_upsample_s, upsample_p=forecaster_upsample_p,
                                rnn_link_size=rnn_link_size, rnn_k=forecaster_rnn_k, rnn_s=forecaster_rnn_s, rnn_p=forecaster_rnn_p, n_layers=forecaster_n_layers,
                                channel_output=forecaster_output, output_k=forecaster_output_k, output_s=forecaster_output_s,
                                output_p=forecaster_output_p, n_output_layers=forecaster_output_layers, batch_norm=batch_norm)
            name = 'Forecaster_' + str(i+1).zfill(2)
            setattr(self, name, model)
            models.append(getattr(self, name))

        self.models = models

    def forward(self, x):
        if x.size()[1] != self.n_encoders:
            assert x.size()[1] == self.n_encoders, '`x` must have the same as n_encoders'

        forecast = []

        for i in range(self.n_encoders):
            if i == 0:
                hidden=None
            model = self.models[i]
            hidden = model(x = x[:,i,:,:,:], hidden=hidden)

        hidden = hidden[::-1]

        for i in range(self.n_encoders,self.n_encoders+self.n_forecasters):

            model = self.models[i]
            hidden, output = model(hidden=hidden)
            forecast.append(output)
        forecast = torch.cat(forecast, dim=1)

        return forecast
