import numpy as numpy
import torch
import torch.nn as nn
import torch.nn.functional as F

from .cnn2D import CNN2D_cell, DeCNN2D_cell

class ConvGRUCell(nn.Module):
    '''
    Generate a convolutional GRU cell
    '''
    def __init__(self, channel_input, channel_hidden, kernel_size, stride, padding, batch_norm=False, device=None, value_dtype=None):
        super().__init__()
        self.device = device
        self.value_dtype = value_dtype
        self.channel_hidden = channel_hidden
        self.reset_gate = CNN2D_cell(channel_input+channel_hidden, channel_hidden, kernel_size, stride=stride, padding=padding, batch_norm=batch_norm)
        self.update_gate = CNN2D_cell(channel_input+channel_hidden, channel_hidden, kernel_size, stride=stride, padding=padding, batch_norm=batch_norm)
        self.out_gate = CNN2D_cell(channel_input+channel_hidden, channel_hidden, kernel_size, stride=stride, padding=padding, batch_norm=batch_norm, negative_slope=0.2)
    
    def forward(self, x, prev_state=None):
        input_ = x
        batch_size = input_.data.shape[0]
        spatial_size = input_.data.shape[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.channel_hidden] + list(spatial_size)
            if torch.cuda.is_available():
                prev_state = torch.zeros(state_size).to(device=self.device, dtype=self.value_dtype)
            else:
                prev_state = torch.zeros(state_size)

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat([input_, prev_state], dim=1)
        update = torch.sigmoid(self.update_gate(stacked_inputs))
        reset = torch.sigmoid(self.reset_gate(stacked_inputs))
        out_inputs = F.leaky_relu(self.out_gate(torch.cat([input_, prev_state*reset], dim=1)), negative_slope=0.2)
        new_state = prev_state*update + out_inputs*(1-update)

        return new_state

class DeConvGRUCell(nn.Module):
    '''
    Generate a convolutional GRU cell
    '''
    def __init__(self, channel_input, channel_hidden, kernel_size, stride, padding, batch_norm=False, device=None, value_dtype=None):
        super().__init__()
        self.device = device
        self.value_dtype = value_dtype
        self.reset_gate = DeCNN2D_cell(channel_input+channel_hidden, channel_hidden, kernel_size, stride=stride, padding=padding, batch_norm=batch_norm)
        self.update_gate = DeCNN2D_cell(channel_input+channel_hidden, channel_hidden, kernel_size, stride=stride, padding=padding, batch_norm=batch_norm)
        self.out_gate = DeCNN2D_cell(channel_input+channel_hidden, channel_hidden, kernel_size, stride=stride, padding=padding, batch_norm=batch_norm, negative_slope=0.2)
    
    def forward(self, x=None, prev_state=None):
        input_ = x
        if input_ is not None:
            batch_size = input_.data.shape[0]
            spatial_size = input_.data.shape[2:]

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


class Encoder(nn.Module):
    def __init__(self, channel_input, channel_downsample, channel_rnn, 
                downsample_k, downsample_s,downsample_p,
                rnn_k, rnn_s, rnn_p, n_layers, 
                batch_norm=False, device=None, value_dtype=None):
        '''
        Generates a multi-layer convolutional GRU.
        Preserves spatial dimensions across cells, only altering depth.
        Parameters
        ----------
        channel_input: (integer.) depth dimension of input tensors.
        channel_downsample: (integer or list.) depth dimensions of downsample.
        channel_rnn: (integer or list.) depth dimensions of rnn.
        downsample_k: (integer or list.) the kernel size of each downsample layers.
        downsample_s: (integer or list.) the stride size of each downsample layers.
        downsample_p: (integer or list.) the padding size of each downsample layers.
        rnn_k: (integer or list.) the kernel size of each rnn layers.
        rnn_s: (integer or list.) the stride size of each rnn layers.
        rnn_p: (integer or list.) the padding size of each rnn layers.
        n_layers: (integer.) number of chained "ConvGRUCell".
        '''
        super().__init__()

        # channel size
        if type(channel_downsample) != list:
            channel_downsample = [channel_downsample]*int(n_layers/2)
        assert len(channel_downsample) == int(n_layers/2), '`channel_downsample` must have the same length as n_layers/2'

        if type(channel_rnn) != list:
            channel_rnn = [channel_rnn]*int(n_layers/2)
        assert len(channel_rnn) == int(n_layers/2), '`channel_rnn` must have the same length as n_layers/2'

        # kernel size
        if type(downsample_k) != list:
            downsample_k = [downsample_k]*int(n_layers/2)
        assert len(downsample_k) == int(n_layers/2), '`downsample_k` must have the same length as n_layers/2'

        if type(rnn_k) != list:
            rnn_k = [rnn_k]*int(n_layers/2)
        assert len(rnn_k) == int(n_layers/2), '`rnn_k` must have the same length as n_layers/2'

       # stride size
        if type(downsample_s) != list:
            downsample_s = [downsample_s]*int(n_layers/2)
        assert len(downsample_s) == int(n_layers/2), '`downsample_s` must have the same length as n_layers/2'

        if type(rnn_s) != list:
            rnn_s = [rnn_s]*int(n_layers/2)
        assert len(rnn_s) == int(n_layers/2), '`rnn_s` must have the same length as n_layers/2'

        # padding size
        if type(downsample_p) != list:
            downsample_p = [downsample_p]*int(n_layers/2)
        assert len(downsample_p) == int(n_layers/2), '`downsample_p` must have the same length as n_layers/2'

        if type(rnn_p) != list:
            rnn_p = [rnn_p]*int(n_layers/2)
        assert len(rnn_p) == int(n_layers/2), '`rnn_p` must have the same length as n_layers/2'

        self.n_layers = n_layers

        cells = []
        for i in range(int(self.n_layers/2)):
            if i == 0:
                cell=CNN2D_cell(channel_input, channel_downsample[i], downsample_k[i],
                                downsample_s[i], downsample_p[i], batch_norm)
            else:
                cell=CNN2D_cell(channel_rnn[i-1], channel_downsample[i], downsample_k[i],
                                downsample_s[i], downsample_p[i], batch_norm)

            name = 'Downsample_' + str(i).zfill(2)
            setattr(self, name, cell)
            cells.append(getattr(self, name))

            cell = ConvGRUCell(channel_downsample[i], channel_rnn[i], rnn_k[i], rnn_s[i], rnn_p[i], 
                                batch_norm=batch_norm, device=device, value_dtype=value_dtype)
            name = 'ConvGRUCell_' + str(i).zfill(2)
            setattr(self, name, cell)
            cells.append(getattr(self, name))

        self.cells = cells

    def forward(self, x, hidden=None):
        if not hidden:
            hidden = [None]*int(self.n_layers/2)

        input_ = x
        upd_hidden = []
        
        for i in range(int(self.n_layers/2)):
            cell = self.cells[2*i]
            input_ = cell(input_)

            cell = self.cells[2*i+1]
            cell_hidden = hidden[i]

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
                rnn_k, rnn_s, rnn_p, n_layers,
                channel_output=1, output_k=1, output_s = 1, output_p=0, n_output_layers=1,
                batch_norm=False, device=None, value_dtype=None):
        '''
        Generates a multi-layer convolutional GRU.
        Preserves spatial dimensions across cells, only altering depth.
        Parameters
        ----------
        channel_input: (integer.) depth dimension of input tensors.
        channel_upsample: (integer or list.) depth dimensions of upsample.
        channel_rnn: (integer or list.) depth dimensions of rnn.
        upsample_k: (integer or list.) the kernel size of upsample layers.
        upsample_s: (integer or list.) the stride size of upsample layers.
        upsample_p: (integer or list.) the padding size of upsample layers.
        rnn_k: (integer or list.) the kernel size of rnn layers.
        rnn_s: (integer or list.) the stride size of rnn layers.
        rnn_p: (integer or list.) the padding size of rnn layers.
        n_layers: (integer.) number of chained "DeConvGRUCell".
        ## output layer params
        channel_output: (integer or list.) depth dimensions of output.
        output_k: (integer or list.) the kernel size of output layers.
        output_s: (integer or list.) the stride size of output layers.
        output_p: (integer or list.) the padding size of output layers.
        n_output_layers=1
        '''
        super().__init__()

        # channel size
        if type(channel_upsample) != list:
            channel_upsample = [channel_upsample]*int(n_layers/2)
        assert len(channel_upsample) == int(n_layers/2), '`channel_upsample` must have the same length as n_layers/2'

        if type(channel_rnn) != list:
            channel_rnn = [channel_rnn]*int(n_layers/2)
        assert len(channel_rnn) == int(n_layers/2), '`channel_rnn` must have the same length as n_layers/2'

        # kernel size
        if type(upsample_k) != list:
            upsample_k = [upsample_k]*int(n_layers/2)
        assert len(upsample_k) == int(n_layers/2), '`upsample_k` must have the same length as n_layers/2'

        if type(rnn_k) != list:
            rnn_k = [rnn_k]*int(n_layers/2)
        assert len(rnn_k) == int(n_layers/2), '`rnn_k` must have the same length as n_layers/2'
        
       # stride size
        if type(upsample_s) != list:
            upsample_s = [upsample_s]*int(n_layers/2)
        assert len(upsample_s) == int(n_layers/2), '`upsample_s` must have the same length as n_layers/2'
        
        if type(rnn_s) != list:
            rnn_s = [rnn_s]*int(n_layers/2)
        assert len(rnn_s) == int(n_layers/2), '`rnn_s` must have the same length as n_layers/2'

        # padding size
        if type(upsample_p) != list:
            upsample_p = [upsample_p]*int(n_layers/2)
        assert len(upsample_p) == int(n_layers/2), '`upsample_p` must have the same length as n_layers/2'

        if type(rnn_p) != list:
            rnn_p = [rnn_p]*int(n_layers/2)
        assert len(rnn_p) == int(n_layers/2), '`rnn_p` must have the same length as n_layers/2'

        # output size
        if type(channel_output) != list:
            channel_output = [channel_output]*int(n_output_layers)
        assert len(channel_output) == int(n_output_layers), '`channel_output` must have the same length as n_output_layers'

        if type(output_k) != list:
            output_k = [output_k]*int(n_output_layers)
        assert len(output_k) == int(n_output_layers), '`output_k` must have the same length as n_output_layers'

        if type(output_p) != list:
            output_p = [output_p]*int(n_output_layers)
        assert len(output_p) == int(n_output_layers), '`output_p` must have the same length as n_output_layers'

        if type(output_s) != list:
            output_s = [output_s]*int(n_output_layers)
        assert len(output_s) == int(n_output_layers), '`output_s` must have the same length as n_output_layers'

        self.n_output_layers = n_output_layers
        self.n_layers = n_layers

        cells = []
        for i in range(int(n_layers/2)):
            if i == 0:
                cell = DeConvGRUCell(channel_input, channel_rnn[i], rnn_k[i], rnn_s[i], rnn_p[i], batch_norm, device, value_dtype)
            else:
                cell = DeConvGRUCell(channel_upsample[i-1], channel_rnn[i], rnn_k[i], rnn_s[i], rnn_p[i], batch_norm, device, value_dtype)

            name = 'DeConvGRUCell_' + str(i).zfill(2)
            setattr(self, name, cell)
            cells.append(getattr(self, name))

            cell = DeCNN2D_cell(channel_rnn[i], channel_upsample[i], upsample_k[i], upsample_s[i], upsample_p[i], batch_norm)
            name = 'Upsample_' + str(i).zfill(2)
            setattr(self, name, cell)
            cells.append(getattr(self, name))

        for i in range(n_output_layers):
            if i == 0:
                cell = CNN2D_cell(channel_upsample[-1], channel_output[i], output_k[i], output_s[i], output_p[i], batch_norm)
            else:
                cell = CNN2D_cell(channel_output[i-1], channel_output[i], output_k[i], output_s[i], output_p[i], batch_norm)
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

        for i in range(int(self.n_layers/2)):
            cell = self.cells[2*i]
            cell_hidden = hidden[i]
            # pass through layer
            upd_cell_hidden = cell(input_, cell_hidden)
            upd_hidden.append(upd_cell_hidden)
            # update input_ to the last updated hidden layer for next pass
            input_ = upd_cell_hidden

            cell = self.cells[2*i+1]
            input_ = cell(input_)
        for i in range(self.n_output_layers):
            cell = self.cells[self.n_layers+i]
            output = cell(input_)

        # retain tensors in list to allow different hidden sizes
        return upd_hidden, output


class Model(nn.Module):
    def __init__(self, n_encoders, n_forecasters,
                encoder_input_channel, encoder_downsample_channels, encoder_rnn_channels,
                encoder_downsample_k, encoder_downsample_s, encoder_downsample_p,
                encoder_rnn_k, encoder_rnn_s, encoder_rnn_p, encoder_n_layers,
                forecaster_input_channel, forecaster_upsample_channels, forecaster_rnn_channels,
                forecaster_upsample_k, forecaster_upsample_s, forecaster_upsample_p,
                forecaster_rnn_k, forecaster_rnn_s, forecaster_rnn_p, forecaster_n_layers,
                forecaster_output=1, forecaster_output_k=1, forecaster_output_s=1, forecaster_output_p=0, forecaster_output_layers=1,
                batch_norm=False, device=None, value_dtype=None):

        super().__init__()
        self.n_encoders = n_encoders
        self.n_forecasters = n_forecasters
        self.name = 'ConvGRU'

        models = []
        for i in range(self.n_encoders):
            model = Encoder(channel_input=encoder_input_channel, channel_downsample=encoder_downsample_channels, channel_rnn=encoder_rnn_channels,
                            downsample_k=encoder_downsample_k, rnn_k=encoder_rnn_k,
                            downsample_s=encoder_downsample_s, rnn_s=encoder_rnn_s,
                            downsample_p=encoder_downsample_p, rnn_p=encoder_rnn_p, n_layers=encoder_n_layers, 
                            batch_norm=batch_norm, device=device, value_dtype=value_dtype)
            name = 'Encoder_' + str(i).zfill(2)
            setattr(self, name, model)
            models.append(getattr(self, name))

        for i in range(self.n_forecasters):
            model=Forecaster(channel_input=forecaster_input_channel, channel_upsample=forecaster_upsample_channels, channel_rnn=forecaster_rnn_channels,
                            upsample_k=forecaster_upsample_k, rnn_k=forecaster_rnn_k,
                            upsample_s=forecaster_upsample_s, rnn_s=forecaster_rnn_s,
                            upsample_p=forecaster_upsample_p, rnn_p=forecaster_rnn_p, n_layers=forecaster_n_layers,
                            channel_output=forecaster_output, output_k=forecaster_output_k, output_s=forecaster_output_s,
                            output_p=forecaster_output_p, n_output_layers=forecaster_output_layers, 
                            batch_norm=batch_norm, device=device, value_dtype=value_dtype)
            name = 'forecaster_' + str(i).zfill(2)
            setattr(self, name, model)
            models.append(getattr(self, name))

        self.models = models

    def forward(self, x):
        if x.shape[1] != self.n_encoders:
            assert x.shape[1] == self.n_encoders, '`x` must have the same as n_encoders'

        forecast = []

        for i in range(self.n_encoders):
            if i == 0:
                hidden=None
            model = self.models[i]
            hidden = model(x = x[:,i,:,:,:], hidden=hidden)

        hidden = hidden[::-1]

        for i in range(self.n_forecasters):
            model = self.models[self.n_encoders+i]
            hidden, output = model(hidden=hidden)
            forecast.append(output)
        forecast = torch.cat(forecast, dim=1)

        return forecast
