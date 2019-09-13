import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .cnn2D import CNN2D_cell, DeCNN2D_cell

class ConvGRUcell(nn.Module):
    '''
    Generate a convolutional GRU cell
    '''
    def __init__(self, channel_input, channel_output, kernel, stride, padding, batch_norm=False):
        super().__init__()
        self.channel_output = channel_output
        self.reset_gate = CNN2D_cell(channel_input+channel_output, channel_output, kernel, stride, padding, batch_norm)
        self.update_gate = CNN2D_cell(channel_input+channel_output, channel_output, kernel, stride, padding, batch_norm)
        self.out_gate = CNN2D_cell(channel_input+channel_output, channel_output, kernel, stride, padding, batch_norm, negative_slope=0.2)

    def forward(self, x, prev_state=None):
        input_ = x
        batch_size = input_.data.shape[0]
        spatial_size = input_.data.shape[2:]

        # get device and dtype
        device = self.reset_gate.layer[0].weight.device
        dtype = self.reset_gate.layer[0].weight.dtype

        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.channel_output] + list(spatial_size)
            prev_state = torch.zeros(state_size).to(device=device, dtype=dtype)

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat([input_, prev_state], dim=1)
        update = torch.sigmoid(self.update_gate(stacked_inputs))
        reset = torch.sigmoid(self.reset_gate(stacked_inputs))
        out_inputs = F.leaky_relu(self.out_gate(torch.cat([input_, prev_state*reset], dim=1)), negative_slope=0.2)
        new_state = prev_state*update + out_inputs*(1-update)

        return new_state

class DeConvGRUcell(nn.Module):
    '''
    Generate a convolutional GRU cell
    '''
    def __init__(self, channel_input, channel_output, kernel, stride, padding, batch_norm=False):
        super().__init__()
        self.reset_gate = CNN2D_cell(channel_input+channel_output, channel_output, kernel, stride, padding, batch_norm)
        self.update_gate = CNN2D_cell(channel_input+channel_output, channel_output, kernel, stride, padding, batch_norm)
        self.out_gate = CNN2D_cell(channel_input+channel_output, channel_output, kernel, stride, padding, batch_norm, negative_slope=0.2)
    
    def forward(self, x=None, prev_state=None):
        input_ = x
        
        # get device and dtype
        device = self.reset_gate.layer[0].weight.device
        dtype = self.reset_gate.layer[0].weight.dtype
        # data size is [batch, channel, height, width]
        if input_ is None:
            stacked_inputs = prev_state
        else:
            stacked_inputs = torch.cat([input_, prev_state], dim=1)
        
        update = torch.sigmoid(self.update_gate(stacked_inputs))
        reset = torch.sigmoid(self.reset_gate(stacked_inputs))

        if input_ is None:
            out_inputs = F.leaky_relu(self.out_gate(prev_state*reset), negative_slope=0.2)
        else:
            out_inputs = F.leaky_relu(self.out_gate(torch.cat([input_, prev_state*reset], dim=1)), negative_slope=0.2)

        new_state = prev_state*(1-update) + out_inputs*update

        return new_state


class Encoder(nn.Module):
    def __init__(self, channel_input, channel_downsample, channel_gru, 
                downsample_k, downsample_s, downsample_p,
                gru_k, gru_s, gru_p, n_cells, 
                batch_norm=False):
        '''
        Generates a multi-layer convolutional GRU.
        Preserves spatial dimensions across cells, only altering depth.
        ----------
        Parameters
        ----------
        channel_input: (integer.) channel size of input tensors.
        channel_downsample: (integer or list.) channel size of downsample layers.
        channel_gru: (integer or list.) channel size of gru layers.
        downsample_k: (integer or list.) kernel size of downsample layers.
        downsample_s: (integer or list.) stride size of downsample layers.
        downsample_p: (integer or list.) padding size of downsample layers.
        gru_k: (integer or list.) kernel size of gru layers.
        gru_s: (integer or list.) stride size of gru layers.
        gru_p: (integer or list.) padding size of gru layers.
        n_cells: (integer.) number of chained "ConvGRUcell".
        '''
        super().__init__()

        # channel size
        if type(channel_downsample) != list:
            channel_downsample = [channel_downsample]*n_cells
        assert len(channel_downsample) == n_cells, '"channel_downsample" must have the same length as n_cells'

        if type(channel_gru) != list:
            channel_gru = [channel_gru]*n_cells
        assert len(channel_gru) == n_cells, '"channel_gru" must have the same length as n_cells'

        # kernel size
        if type(downsample_k) != list:
            downsample_k = [downsample_k]*n_cells
        assert len(downsample_k) == n_cells, '"downsample_k" must have the same length as n_cells'
       # stride size
        if type(downsample_s) != list:
            downsample_s = [downsample_s]*n_cells
        assert len(downsample_s) == n_cells, '"downsample_s" must have the same length as n_cells'
        # padding size
        if type(downsample_p) != list:
            downsample_p = [downsample_p]*n_cells
        assert len(downsample_p) == n_cells, '"downsample_p" must have the same length as n_cells'

        if type(gru_k) != list:
            gru_k = [gru_k]*n_cells
        assert len(gru_k) == n_cells, '"gru_k" must have the same length as n_cells'
        if type(gru_s) != list:
            gru_s = [gru_s]*n_cells
        assert len(gru_s) == n_cells, '"gru_s" must have the same length as n_cells'
        if type(gru_p) != list:
            gru_p = [gru_p]*n_cells
        assert len(gru_p) == n_cells, '"gru_p" must have the same length as n_cells'

        self.n_cells = n_cells

        cells = []
        for i in range(n_cells):
            if i == 0:
                cell = CNN2D_cell(channel_input, channel_downsample[i], downsample_k[i], downsample_s[i], downsample_p[i], batch_norm)
            else:
                cell = CNN2D_cell(channel_gru[i-1], channel_downsample[i], downsample_k[i], downsample_s[i], downsample_p[i], batch_norm)

            name = 'Downsample_' + str(i).zfill(2)
            setattr(self, name, cell)
            cells.append(getattr(self, name))

            cell = ConvGRUcell(channel_downsample[i], channel_gru[i], gru_k[i], gru_s[i], gru_p[i], batch_norm)
            name = 'ConvGRUcell_' + str(i).zfill(2)
            setattr(self, name, cell)
            cells.append(getattr(self, name))

        self.cells = cells

    def forward(self, x, hidden=None):
        if hidden is None:
            hidden = [None]*int(self.n_cells)

        input_ = x
        upd_hidden = []
        
        for i in range(self.n_cells):
            # CNN layer
            cell = self.cells[2*i]
            input_ = cell(input_)
            # ConvGRUcell layer
            cell = self.cells[2*i+1]
            upd_cell_hidden = cell(input_, hidden[i])
            upd_hidden.append(upd_cell_hidden)
            # new hidden state as input for the next CNN layer
            input_ = upd_cell_hidden

        # return new hidden state (list.)
        return upd_hidden

class Forecaster(nn.Module):
    def __init__(self, channel_input, channel_upsample, channel_gru,
                upsample_k, upsample_p, upsample_s,
                gru_k, gru_s, gru_p, n_cells,
                channel_output=1, output_k=1, output_s = 1, output_p=0, n_output_layers=1,
                batch_norm=False):
        '''
        Generates a multi-layer convolutional GRU.
        Preserves spatial dimensions across cells, only altering depth.
        Parameters
        ----------
        channel_input: (integer.) channel size of input tensors.
        channel_upsample: (integer or list.) output channel sizes of upsample layers.
        channel_gru: (integer or list.) output channel size of gru cells.
        upsample_k: (integer or list.) kernel size of upsample layers.
        upsample_s: (integer or list.) stride size of upsample layers.
        upsample_p: (integer or list.) padding size of upsample layers.
        gru_k: (integer or list.) kernel size of gru layers.
        gru_s: (integer or list.) stride size of gru layers.
        gru_p: (integer or list.) padding size of gru layers.
        n_cells: (integer.) number of chained "DeConvGRUcell".
        # output layer params
        channel_output: (integer or list.) output channel size of output layer.
        output_k: (integer or list.) kernel size of output layers.
        output_s: (integer or list.) stride size of output layers.
        output_p: (integer or list.) padding size of output layers.
        n_output_layers=1
        '''
        super().__init__()
        # channel size
        if type(channel_upsample) != list:
            channel_upsample = [channel_upsample]*n_cells
        assert len(channel_upsample) == n_cells, '"channel_upsample" must have the same length as n_cells'

        if type(channel_gru) != list:
            channel_gru = [channel_gru]*n_cells
        assert len(channel_gru) == n_cells, '"channel_gru" must have the same length as n_cells'

        # kernel size
        if type(upsample_k) != list:
            upsample_k = [upsample_k]*n_cells
        assert len(upsample_k) == n_cells, '"upsample_k" must have the same length as n_cells'

        if type(gru_k) != list:
            gru_k = [gru_k]*n_cells
        assert len(gru_k) == n_cells, '"gru_k" must have the same length as n_cells'
        
       # stride size
        if type(upsample_s) != list:
            upsample_s = [upsample_s]*n_cells
        assert len(upsample_s) == n_cells, '"upsample_s" must have the same length as n_cells'
        
        if type(gru_s) != list:
            gru_s = [gru_s]*n_cells
        assert len(gru_s) == n_cells, '"gru_s" must have the same length as n_cells'

        # padding size
        if type(upsample_p) != list:
            upsample_p = [upsample_p]*n_cells
        assert len(upsample_p) == n_cells, '"upsample_p" must have the same length as n_cells'

        if type(gru_p) != list:
            gru_p = [gru_p]*n_cells
        assert len(gru_p) == n_cells, '"gru_p" must have the same length as n_cells'

        # output size
        if type(channel_output) != list:
            channel_output = [channel_output]*n_output_layers
        assert len(channel_output) == n_output_layers, '"channel_output" must have the same length as n_output_layers'

        if type(output_k) != list:
            output_k = [output_k]*n_output_layers
        assert len(output_k) == n_output_layers, '"output_k" must have the same length as n_output_layers'

        if type(output_p) != list:
            output_p = [output_p]*n_output_layers
        assert len(output_p) == n_output_layers, '"output_p" must have the same length as n_output_layers'

        if type(output_s) != list:
            output_s = [output_s]*n_output_layers
        assert len(output_s) == n_output_layers, '"output_s" must have the same length as n_output_layers'

        self.n_cells = n_cells
        self.n_output_layers = n_output_layers

        cells = []
        for i in range(n_cells):
            if i == 0:
                cell = DeConvGRUcell(channel_input, channel_gru[i], gru_k[i], gru_s[i], gru_p[i], batch_norm)
            else:
                cell = DeConvGRUcell(channel_upsample[i-1], channel_gru[i], gru_k[i], gru_s[i], gru_p[i], batch_norm)

            name = 'DeConvGRUcell_' + str(i).zfill(2)
            setattr(self, name, cell)
            cells.append(getattr(self, name))

            cell = DeCNN2D_cell(channel_gru[i], channel_upsample[i], upsample_k[i], upsample_s[i], upsample_p[i], batch_norm)
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

        for i in range(int(self.n_cells)):
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
            cell = self.cells[2*self.n_cells+i]
            output = cell(input_)

        # retain tensors in list to allow different hidden sizes
        return upd_hidden, output

class Model(nn.Module):
    def __init__(self, n_encoders, n_forecasters,
                encoder_input_channel, encoder_downsample_channels, encoder_gru_channels,
                encoder_downsample_k, encoder_downsample_s, encoder_downsample_p,
                encoder_gru_k, encoder_gru_s, encoder_gru_p, encoder_n_cells,
                forecaster_input_channel, forecaster_upsample_channels, forecaster_gru_channels,
                forecaster_upsample_k, forecaster_upsample_s, forecaster_upsample_p,
                forecaster_gru_k, forecaster_gru_s, forecaster_gru_p, forecaster_n_cells,
                forecaster_output=1, forecaster_output_k=1, forecaster_output_s=1, forecaster_output_p=0, forecaster_output_layers=1, 
                batch_norm=False, target_RAD=False):

        super().__init__()
        self.n_encoders = n_encoders
        self.n_forecasters = n_forecasters
        self.name = 'CONVGRU'
        self.target_RAD = target_RAD

        self.encoder = Encoder(channel_input=encoder_input_channel, channel_downsample=encoder_downsample_channels, channel_gru=encoder_gru_channels,
                                downsample_k=encoder_downsample_k, gru_k=encoder_gru_k,
                                downsample_s=encoder_downsample_s, gru_s=encoder_gru_s,
                                downsample_p=encoder_downsample_p, gru_p=encoder_gru_p, n_cells=encoder_n_cells, 
                                batch_norm=batch_norm)

        self.forecaster = Forecaster(channel_input=forecaster_input_channel, channel_upsample=forecaster_upsample_channels, channel_gru=forecaster_gru_channels,
                                    upsample_k=forecaster_upsample_k, gru_k=forecaster_gru_k,
                                    upsample_s=forecaster_upsample_s, gru_s=forecaster_gru_s,
                                    upsample_p=forecaster_upsample_p, gru_p=forecaster_gru_p, n_cells=forecaster_n_cells,
                                    channel_output=forecaster_output, output_k=forecaster_output_k, output_s=forecaster_output_s,
                                    output_p=forecaster_output_p, n_output_layers=forecaster_output_layers, 
                                    batch_norm=batch_norm)

    def forward(self, x):
        if x.shape[1] != self.n_encoders:
            assert x.shape[1] == self.n_encoders, '"x" must have the same as n_encoders'

        hidden = None

        for i in range(self.n_encoders):
            hidden = self.encoder(x = x[:,i,:,:,:], hidden=hidden)

        hidden = hidden[::-1]

        forecast = []
        for i in range(self.n_forecasters):
            hidden, output = self.forecaster(hidden=hidden)
            forecast.append(output)

        forecast = torch.cat(forecast, dim=1)
        if not self.target_RAD:
            forecast = ((10**(forecast/10))/200)**(5/8)
        return forecast

class Multi_unit_Model(nn.Module):
    def __init__(self, n_encoders, n_forecasters,
                encoder_input_channel, encoder_downsample_channels, encoder_gru_channels,
                encoder_downsample_k, encoder_downsample_s, encoder_downsample_p,
                encoder_gru_k, encoder_gru_s, encoder_gru_p, encoder_n_cells,
                forecaster_input_channel, forecaster_upsample_channels, forecaster_gru_channels,
                forecaster_upsample_k, forecaster_upsample_s, forecaster_upsample_p,
                forecaster_gru_k, forecaster_gru_s, forecaster_gru_p, forecaster_n_cells,
                forecaster_output=1, forecaster_output_k=1, forecaster_output_s=1, forecaster_output_p=0, forecaster_output_layers=1,
                batch_norm=False, target_RAD=False):

        super().__init__()
        self.n_encoders = n_encoders
        self.n_forecasters = n_forecasters
        self.name = 'Multi_unit_CONVGRU'
        self.target_RAD = target_RAD

        models = []
        for i in range(self.n_encoders):
            model = Encoder(channel_input=encoder_input_channel, channel_downsample=encoder_downsample_channels, channel_gru=encoder_gru_channels,
                            downsample_k=encoder_downsample_k, gru_k=encoder_gru_k,
                            downsample_s=encoder_downsample_s, gru_s=encoder_gru_s,
                            downsample_p=encoder_downsample_p, gru_p=encoder_gru_p, n_cells=encoder_n_cells, 
                            batch_norm=batch_norm)
            name = 'Encoder_' + str(i).zfill(2)
            setattr(self, name, model)
            models.append(getattr(self, name))

        for i in range(self.n_forecasters):
            model=Forecaster(channel_input=forecaster_input_channel, channel_upsample=forecaster_upsample_channels, channel_gru=forecaster_gru_channels,
                            upsample_k=forecaster_upsample_k, gru_k=forecaster_gru_k,
                            upsample_s=forecaster_upsample_s, gru_s=forecaster_gru_s,
                            upsample_p=forecaster_upsample_p, gru_p=forecaster_gru_p, n_cells=forecaster_n_cells,
                            channel_output=forecaster_output, output_k=forecaster_output_k, output_s=forecaster_output_s,
                            output_p=forecaster_output_p, n_output_layers=forecaster_output_layers, 
                            batch_norm=batch_norm)
            name = 'forecaster_' + str(i).zfill(2)
            setattr(self, name, model)
            models.append(getattr(self, name))

        self.models = models

    def forward(self, x):
        if x.shape[1] != self.n_encoders:
            assert x.shape[1] == self.n_encoders, '"x" must have the same as n_encoders'

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

        if not self.target_RAD:
            forecast = ((10**(forecast/10))/200)**(5/8)

        return forecast
