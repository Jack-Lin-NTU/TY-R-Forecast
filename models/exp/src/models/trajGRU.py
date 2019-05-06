import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .cnn2D import CNN2D_cell, DeCNN2D_cell

class flow_warp(nn.Module):
    '''
    Arguments:
        The subcnn model and the M warp function 
    '''
    def __init__(self, channel_input, channel_hidden, link_size, kernel_size=1, stride=1, padding=0, batch_norm=False, device=None, value_dtype=None):
        super().__init__()
        self.device = device
        self.value_dtype = value_dtype
        self.channel_input = channel_input
        self.channel_hidden = channel_hidden
        self.link_size = link_size
        # 2 cnn layers
        displacement_layers = []
        displacement_layers.append(nn.Conv2d(channel_input+channel_hidden, 32, 5, 1, 2))
        displacement_layers.append(nn.LeakyReLU(negative_slope=0.2))
        displacement_layers.append(nn.Conv2d(32, link_size*2, 5, 1, 2))
        displacement_layers.append(nn.LeakyReLU(negative_slope=0.2))

        # initialize the weightings in each layers.
        # nn.nn.init.orthogonal_(displacement_layers[0].weight)

        nn.init.kaiming_normal_(displacement_layers[0].weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(displacement_layers[2].weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
        # nn.init.zeros_(displacement_layers[0].weight)
        # nn.init.zeros_(displacement_layers[2].weight)
        nn.init.zeros_(displacement_layers[0].bias)
        nn.init.zeros_(displacement_layers[2].bias)
        self.displacement_layers = nn.Sequential(*displacement_layers)

    def grid_sample(self, x, flow):
        '''
        Function for sampling pixels based on given grid data.
        '''
        input_ = x
        b, _, h, w = input_.shape
        u, v = flow[:,0:self.link_size,:,:], flow[:,self.link_size:,:,:]
        samples = []
        for i in range(self.link_size):
            y_, x_ = torch.meshgrid(torch.arange(h), torch.arange(w))
            y_, x_ = y_.expand(b,-1,-1).to(self.device, dtype=self.value_dtype), x_.expand(b,-1,-1).to(self.device, dtype=self.value_dtype)
            x_ = (x_*2/w)-1 + u[:,i,:,:]
            y_ = (y_*2/w)-1 + v[:,i,:,:]
            grids = torch.stack([x_, y_], 3)
            samples.append(F.grid_sample(input_, grids))
        return torch.cat(samples, dim=1)

    def forward(self, x=None, prev_state=None):
        # get batch and spatial sizes
        # print('Prev:', prev_state.shape)
        input_ = x
        
        if input_ is None:
            stacked_inputs = prev_state
        else:
            stacked_inputs = torch.cat([input_, prev_state], dim=1)

        output = self.displacement_layers(stacked_inputs)
        output = self.grid_sample(x=prev_state, flow=output)

        return output

class TrajGRUCell(nn.Module):
    """
    Arguments: 
        This class is to generate a convolutional Traj_GRU cell.
    """
    def __init__(self, channel_input, channel_hidden, link_size, kernel_size, stride=1, padding=1, batch_norm=False, device=None, value_dtype=None):
        super().__init__()
        self.device = device
        self.value_dtype = value_dtype
        self.channel_input = channel_input
        self.channel_hidden = channel_hidden
        self.link_size = link_size

        self.reset_gate= CNN2D_cell(channel_input+channel_hidden*link_size, channel_hidden, kernel_size, stride, padding, batch_norm=batch_norm)
        self.update_gate = CNN2D_cell(channel_input+channel_hidden*link_size, channel_hidden, kernel_size, stride, padding, batch_norm=batch_norm)
        self.out_gate = CNN2D_cell(channel_input+channel_hidden*link_size, channel_hidden, kernel_size, stride, padding, batch_norm=batch_norm, negative_slope=0.2)

        self.flow_warp = flow_warp(channel_input, channel_hidden, link_size, 1, 1, 0, batch_norm, device, value_dtype)

    def forward(self, x=None, prev_state=None):
        input_ = x

        # get batch and spatial sizes
        batch_size = input_.data.shape[0]
        H, W = input_.data.shape[2:]
        
        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = (batch_size, self.channel_hidden, H, W)
            if torch.cuda.is_available():
                prev_state = torch.zeros(state_size).to(device=self.device, dtype=self.value_dtype)
            else:
                prev_state = torch.zeros(state_size)

        M = self.flow_warp(x=input_, prev_state=prev_state)
        stack_inputs = torch.cat([input_, M], dim=1)
        
        # data size is [batch, channel, height, width]
        reset = torch.sigmoid(self.reset_gate(stack_inputs)).unsqueeze(1).expand(-1,self.link_size,-1,-1,-1).reshape(batch_size,self.link_size*self.channel_hidden,H,W)
        update = torch.sigmoid(self.update_gate(stack_inputs))
        out_inputs = F.leaky_relu(self.out_gate(torch.cat([input_, M*reset], dim=1)), negative_slope=0.2)
        new_state = prev_state*update + out_inputs*(1-update)

        return new_state

class DeTrajGRUCell(nn.Module):
    """
    Arguments: 
        This class is to generate a deconvolutional Traj_GRU cell.
    """
    def __init__(self, channel_input, channel_hidden, link_size, kernel_size, stride=1, padding=1, batch_norm=False, device=None, value_dtype=None):
        super().__init__()
        self.device = device
        self.value_dtype = value_dtype
        self.channel_input = channel_input
        self.channel_hidden = channel_hidden
        self.link_size = link_size
        
        self.reset_gate = DeCNN2D_cell(channel_input+channel_hidden*link_size, channel_hidden, kernel_size, stride, padding, batch_norm)
        self.update_gate = DeCNN2D_cell(channel_input+channel_hidden*link_size, channel_hidden, kernel_size, stride, padding, batch_norm)
        self.out_gate = DeCNN2D_cell(channel_input+channel_hidden*link_size, channel_hidden, kernel_size, stride, padding, batch_norm, negative_slope=0.2)
        self.flow_warp = flow_warp(channel_input, channel_hidden, link_size, 1, 1, 0, batch_norm, device, value_dtype)

    def forward(self, x=None, prev_state=None):
        input_ = x
        # get batch and spatial sizes
        batch_size = prev_state.data.shape[0]
        H, W = prev_state.data.shape[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = (batch_size, self.channel_hidden, H, W)
            if torch.cuda.is_available():
                prev_state = torch.zeros(state_size).to(device=self.device, dtype=self.value_dtype)
            else:
                prev_state = torch.zeros(state_size)

        M = self.flow_warp(x=input_, prev_state=prev_state)

        if self.channel_input == 0:
            reset = torch.sigmoid(self.reset_gate(M)).unsqueeze(1).expand(-1,self.link_size,-1,-1,-1).reshape(batch_size,self.link_size*self.channel_hidden,H,W)
            update = torch.sigmoid(self.update_gate(M))
            out_inputs = F.leaky_relu(self.out_gate(M*reset), negative_slope=0.2)
        else:
            stack_inputs = torch.cat([input_, M], dim=1)
            reset = torch.sigmoid(self.reset_gate(stack_inputs)).unsqueeze(1).expand(-1,self.link_size,-1,-1,-1).reshape(batch_size,self.link_size*self.channel_hidden,H,W)
            update = torch.sigmoid(self.update_gate(stack_inputs))
            out_inputs = F.leaky_relu(self.out_gate(torch.cat([input_, M*reset], dim=1)), negative_slope=0.2)
        
        new_state = prev_state*(1-update) + out_inputs*update
        return new_state

class Encoder(nn.Module):
    def __init__(self, channel_input, channel_downsample, channel_rnn, downsample_k, downsample_s, downsample_p,
                 rnn_link_size, rnn_k, rnn_s, rnn_p, n_layers, batch_norm=False, device=None, value_dtype=None):
        '''
        Argumensts:
            Generates a multi-layer convolutional GRU, which is called encoder.
            Preserves spatial dimensions across cells, only altering depth.
            [Parameters]
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
        self.device = device
        self.value_dtype = value_dtype
        self.channel_input = channel_input

    ## set self variables  
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

    ## set encoder
        cells = []
        for i in range(int(self.n_layers/2)):
            ## convolution cell
            if i == 0:
                cell=CNN2D_cell(channel_input=self.channel_input, channel_hidden=self.channel_downsample[i], kernel_size=self.downsample_k[i], 
                                stride=self.downsample_s[i], padding=self.downsample_p[i], batch_norm=batch_norm)
            else:
                cell=CNN2D_cell(channel_input=self.channel_rnn[i-1], channel_hidden=self.channel_downsample[i], kernel_size=self.downsample_k[i], 
                                stride=self.downsample_s[i], padding=self.downsample_p[i], batch_norm=batch_norm)

            name = 'Downsample_' + str(i).zfill(2)
            setattr(self, name, cell)
            cells.append(getattr(self, name))
            
            ## gru cell
            cell = TrajGRUCell(channel_input=self.channel_downsample[i], channel_hidden=self.channel_rnn[i], link_size=self.rnn_link_size[i],
                               kernel_size=self.rnn_k[i], stride=self.rnn_s[i], padding=self.rnn_p[i], device=self.device, value_dtype=self.value_dtype)
            name = 'TrajGRUCell_' + str(i).zfill(2)
            setattr(self, name, cell)
            cells.append(getattr(self, name))

        self.cells = cells
    
    def forward(self, x=None, hidden=None):
        if hidden is None:
            hidden = [None]*int(self.n_layers/2)

        input_ = x
        upd_hidden = []

        for i in range(int(self.n_layers/2)):
            ## Convolution cell
            cell = self.cells[2*i]
            input_ = cell(input_)
            ## GRU cell
            cell = self.cells[2*i+1]
            cell_hidden = hidden[i]
            # TrajGRUcell(x=None, prev_state=None)
            upd_cell_hidden = cell(x=input_, prev_state=cell_hidden)
            upd_hidden.append(upd_cell_hidden)
            # Pass input_ to the next
            input_ = upd_cell_hidden
        # retain tensors in list to allow different hidden sizes
        return upd_hidden

class Forecaster(nn.Module):
    '''
        Argumensts:
            Generates a multi-layer deconvolutional GRU, which is called forecaster.
            Preserves spatial dimensions across cells, only altering depth.
            [Parameters]
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
    def __init__(self, channel_input, channel_upsample, channel_rnn, upsample_k, upsample_p, upsample_s,
                 rnn_link_size, rnn_k, rnn_s, rnn_p, n_layers, channel_output=1, output_k=1, output_s = 1, 
                 output_p=0, n_output_layers=1, batch_norm=False, device=None, value_dtype=None):
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

    ## set self variables  
        self.device = device
        self.value_dtype = value_dtype
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

    ## set forecaster
        cells = []
        for i in range(int(self.n_layers/2)):
            # deTraj gru
            if i == 0:
                cell = DeTrajGRUCell(channel_input=self.channel_input, channel_hidden=self.channel_rnn[i], link_size=self.rnn_link_size[i],
                                     kernel_size=self.rnn_k[i], stride=self.rnn_s[i], padding=self.rnn_p[i], device=self.device, value_dtype=self.value_dtype)
            else:
                cell = DeTrajGRUCell(channel_input=self.channel_upsample[i-1], channel_hidden=self.channel_rnn[i], link_size=self.rnn_link_size[i],
                                     kernel_size=self.rnn_k[i], stride=self.rnn_s[i], padding=self.rnn_p[i], device=self.device, value_dtype=self.value_dtype)

            name = 'DeTrajGRUCelll_' + str(i).zfill(2)
            setattr(self, name, cell)
            cells.append(getattr(self, name))
            # decon  
            cell = DeCNN2D_cell(self.channel_rnn[i], self.channel_upsample[i], self.upsample_k[i], self.upsample_s[i], self.upsample_p[i], batch_norm=batch_norm)
            name = 'Upsample_' + str(i).zfill(2)
            setattr(self, name, cell)
            cells.append(getattr(self, name))
        # output layer
        for i in range(self.n_output_layers):
            if i == 0:
                cell = CNN2D_cell(self.channel_upsample[-1], self.channel_output[i], self.output_k[i], self.output_s[i], self.output_p[i])
            else:
                cell = CNN2D_cell(self.channel_output[i-1], self.channel_output[i], self.output_k[i], self.output_s[i], self.output_p[i])
        
            name = 'OutputLayer_' + str(i).zfill(2)
            setattr(self, name, cell)
            cells.append(getattr(self, name))
        
        self.cells = cells

    def forward(self, hidden=None):
        '''
        Parameters
        ----------
        x : 4D input tensor. (batch, channels, height, width).
        hidden : list of 4D hidden state representations. (batch, channels, height, width).
        Returns
        -------
        upd_hidden : 5D hidden representation. (layer, batch, channels, height, width).
        '''

        upd_hidden = []
        output = 0

        for i in range(int(self.n_layers/2)):
            if i == 0:
                ## Top gru cell in forecaster, no need the inputs
                cell = self.cells[2*i]
                cell_hidden = hidden[i]
                # pass through layer
                upd_cell_hidden = cell(prev_state=cell_hidden)
                upd_hidden.append(upd_cell_hidden)
            else:
                ## other gru cells in forecaster, need the inputs
                cell = self.cells[2*i]
                cell_hidden = hidden[i]
                # pass through layer
                upd_cell_hidden = cell(x=input_, prev_state=cell_hidden)
                upd_hidden.append(upd_cell_hidden)
            
            # update input_ to the last updated hidden layer for next pass
            input_ = upd_cell_hidden
            ## deconvolution
            cell = self.cells[2*i+1]
            input_ = cell(upd_cell_hidden)

        ## output layer
        cell = self.cells[-1]
        output = cell(input_)
        ## transfer rad to qpe
        output = ((10**(output/10))/200)**(5/8)
        # retain tensors in list to allow different hidden sizes
        return upd_hidden, output

class Model(nn.Module):
    '''
        Argumensts:
            This class is used to construt whole TrajGRU model based on given parameters.
        '''
    def __init__(self, n_encoders, n_forecasters, rnn_link_size,
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
        self.name = 'TrajGRU'

        models = []
        # encoders
        for i in range(self.n_encoders):
            model = Encoder(channel_input=encoder_input_channel, channel_downsample=encoder_downsample_channels,
                            channel_rnn=encoder_rnn_channels, downsample_k=encoder_downsample_k, downsample_s=encoder_downsample_s, 
                            downsample_p=encoder_downsample_p, rnn_link_size=rnn_link_size, rnn_k=encoder_rnn_k, rnn_s=encoder_rnn_s, 
                            rnn_p=encoder_rnn_p, n_layers=encoder_n_layers, batch_norm=batch_norm, device=device, value_dtype=value_dtype)
            name = 'Encoder_' + str(i).zfill(2)
            setattr(self, name, model)
            models.append(getattr(self, name))

        # forecasters
        for i in range(self.n_forecasters):
            model = Forecaster(channel_input=forecaster_input_channel, channel_upsample=forecaster_upsample_channels, 
                               channel_rnn=forecaster_rnn_channels, upsample_k=forecaster_upsample_k, upsample_s=forecaster_upsample_s, 
                               upsample_p=forecaster_upsample_p, rnn_link_size=rnn_link_size, rnn_k=forecaster_rnn_k, 
                               rnn_s=forecaster_rnn_s, rnn_p=forecaster_rnn_p, n_layers=forecaster_n_layers,
                               channel_output=forecaster_output, output_k=forecaster_output_k, output_s=forecaster_output_s,
                               output_p=forecaster_output_p, n_output_layers=forecaster_output_layers, batch_norm=batch_norm, 
                               device=device, value_dtype=value_dtype)
            name = 'Forecaster_' + str(i).zfill(2)
            setattr(self, name, model)
            models.append(getattr(self, name))

        self.models = models

        # set flow_warp
        encoder_flow_warp = []
        forecaster_flow_warp = []
        for i in range(int(encoder_n_layers/2)):
            model = flow_warp(encoder_downsample_channels[i], encoder_rnn_channels[i], rnn_link_size[i], 1, 1, 0, device, value_dtype)
            name = 'Encoder_flow_warp_' + str(i).zfill(2)
            setattr(self, name, model)
            encoder_flow_warp.append(getattr(self, name))

        for i in range(int(forecaster_n_layers/2)):
            if i == 0:
                model = flow_warp(forecaster_input_channel, forecaster_rnn_channels[i], rnn_link_size[i], 1, 1, 0, device, value_dtype)
            else:
                model = flow_warp(forecaster_upsample_channels[i-1], forecaster_rnn_channels[i], rnn_link_size[i], 1, 1, 0, device, value_dtype)
            name = 'Forecaster_flow_warp_' + str(i).zfill(2)
            setattr(self, name, model)
            forecaster_flow_warp.append(getattr(self, name))

        self.encoder_flow_warp = encoder_flow_warp
        self.forecaster_flow_warp = forecaster_flow_warp

    def forward(self, x):
        input_ = x
        if input_.size()[1] != self.n_encoders:
            assert input_.size()[1] == self.n_encoders, '`x` must have the same as n_encoders'

        forecast = []

        for i in range(self.n_encoders):
            if i == 0:
                hidden=None
            model = self.models[i]
            hidden = model(x = input_[:,i,:,:,:], flow_warp=self.encoder_flow_warp, hidden=hidden)

        hidden = hidden[::-1]

        for i in range(self.n_forecasters):
            model = self.models[self.n_encoders+i]
            hidden, output = model(flow_warp=self.forecaster_flow_warp, hidden=hidden)
            forecast.append(output)
            
        forecast = torch.cat(forecast, dim=1)
        
        return forecast
