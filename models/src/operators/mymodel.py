import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .cnn2D import CNN2D_cell
from .convGRU import ConvGRUCell, DeCNN2D_cell, Encoder

class TyCatcher(nn.Module):
    def __init__(self, channel_input, channel_hidden, n_layers, device=None, value_dtype=None):
        super().__init__()
        self.device = device
        self.value_dtype = value_dtype

        if type(channel_hidden) != list:
            channel_hidden = [channel_hidden]*int(n_layers)
        else:
            assert len(channel_hidden) == n_layers, 'The length of "Channel_hidden" should be same as n_layers'

        layers = []
        for i in range(n_layers):
            if i == 0:
                layer = nn.Linear(channel_input, channel_hidden[i])
            else:
                layer = nn.Linear(channel_hidden[i-1], channel_hidden[i])
            
            nn.init.kaiming_normal_(layer.weight)
            nn.init.constant_(layer.bias, 0.)

            name = 'Layer1_' + str(i).zfill(2)
            setattr(self, name, layer)
            layers.append(getattr(self, name))

        self.n_layers = n_layers
        self.layers = layers
    
    def forward(self, ty_info, rader_map):
        b, c, _, _ = rader_map.shape
        # feed ty info to get theta
        output = ty_info
        for i in range(self.n_layers):
            layer = self.layers[i]
            output = layer(output)
        # breakpoint()
        output = output.view(-1,2,3)
        theta = torch.tensor([[1, 0, 0],[0, 1, 0]]).to(device=self.device, dtype=self.value_dtype).expand(b,2,3) + output
        size = torch.Size((b,c,400,400))
        flowfield = F.affine_grid(theta, size)
        sample = F.grid_sample(rader_map, flowfield)
        return sample

class my_single_GRU(nn.Module):
    def __init__(self, input_frames, output_frames, TyCatcher_channel_input, TyCatcher_channel_hidden, TyCatcher_channel_n_layers, gru_channel_input, gru_channel_hidden, 
                gru_kernel, gru_stride, gru_padding, batch_norm=False, device=None, value_dtype=None):
        super().__init__()
        self.device = device
        self.value_dtype = value_dtype
        self.input_frames = input_frames
        self.output_frames = output_frames

        self.tycatcher = TyCatcher(TyCatcher_channel_input, TyCatcher_channel_hidden, TyCatcher_channel_n_layers, device, value_dtype)
        self.model = ConvGRUCell(gru_channel_input, gru_channel_hidden, gru_kernel, gru_stride, gru_padding, batch_norm, device, value_dtype)

    def forward(self, ty_infos, radar_map):
        
        for i in range(self.input_frames):
            tmp_ty_info = ty_infos[:,i,:]
            tmp_map = radar_map[:,i,:,:,:]
            if i == 0:
                input_ = self.tycatcher(tmp_ty_info, tmp_map)
                prev_state = self.model(input_, prev_state=None)
            else:
                input_ = self.tycatcher(tmp_ty_info, tmp_map)
                prev_state = self.model(input_)
        
        outputs = []
        for i in range(self.output_frames):
            tmp_ty_info = ty_infos[:,i+self.input_frames,:]
            tmp_map = radar_map[:,-1,:,:,:]
            input_ = self.ty_catcher(tmp_ty_info, tmp_map)
            prev_state = self.model(input_)

            outputs.append(prev_state)
        outputs = torch.cat(outputs, dim=1)
        return outputs

class Forecaster(nn.Module):
    def __init__(self, upsample_cin, upsample_cout, upsample_k, upsample_p, upsample_s, n_layers, 
                output_cout=1, output_k=1, output_s=1, output_p=0, n_output_layers=1,
                batch_norm=False, device=None, value_dtype=None):

        super().__init__()
        # channel size
        if type(upsample_cin) != list:
            upsample_cin = [upsample_cin]*int(n_layers)
        assert len(upsample_cin) == int(n_layers), '"upsample_cin" must have the same length as n_layers'

        if type(upsample_cout) != list:
            upsample_cout = [upsample_cout]*int(n_layers)
        assert len(upsample_cout) == int(n_layers), '"upsample_cout" must have the same length as n_layers'

        # kernel size
        if type(upsample_k) != list:
            upsample_k = [upsample_k]*int(n_layers)
        assert len(upsample_k) == int(n_layers), '"upsample_k" must have the same length as n_layers'
        
       # stride size
        if type(upsample_s) != list:
            upsample_s = [upsample_s]*int(n_layers)
        assert len(upsample_s) == int(n_layers), '"upsample_s" must have the same length as n_layers'

        # padding size
        if type(upsample_p) != list:
            upsample_p = [upsample_p]*int(n_layers)
        assert len(upsample_p) == int(n_layers), '"upsample_p" must have the same length as n_layers'

        # output size
        if type(output_cout) != list:
            output_cout = [output_cout]*int(n_output_layers)
        assert len(output_cout) == int(n_output_layers), '"output_cout" must have the same length as n_output_layers'

        if type(output_k) != list:
            output_k = [output_k]*int(n_output_layers)
        assert len(output_k) == int(n_output_layers), '"output_k" must have the same length as n_output_layers'

        if type(output_p) != list:
            output_p = [output_p]*int(n_output_layers)
        assert len(output_p) == int(n_output_layers), '"output_p" must have the same length as n_output_layers'

        if type(output_s) != list:
            output_s = [output_s]*int(n_output_layers)
        assert len(output_s) == int(n_output_layers), '"output_s" must have the same length as n_output_layers'

        self.n_output_layers = n_output_layers
        self.n_layers = n_layers

        cells = []
        for i in range(n_layers):
            if i == 0:
                cell = DeCNN2D_cell(upsample_cin[i], upsample_cout[i], upsample_k[i], upsample_s[i], upsample_p[i], batch_norm)
            else:
                cell = DeCNN2D_cell(upsample_cin[i]+upsample_cout[i-1], upsample_cout[i], upsample_k[i], upsample_s[i], upsample_p[i], batch_norm)

            name = 'Upsample_' + str(i).zfill(2)
            setattr(self, name, cell)
            cells.append(getattr(self, name))

        for i in range(n_output_layers):
            if i == 0:
                cell = CNN2D_cell(upsample_cout[-1], output_cout[i], output_k[i], output_s[i], output_p[i], batch_norm)
            else:
                cell = CNN2D_cell(output_cout[i-1], output_cout[i], output_k[i], output_s[i], output_p[i], batch_norm)
            name = 'OutputLayer_' + str(i).zfill(2)
            setattr(self, name, cell)
            cells.append(getattr(self, name))
            self.cells = cells

    def forward(self, x):
        '''
        Parameters
        ----------
        x : 4D input tensor. (batch, channels, height, width).
        hidden : list of 4D hidden state representations. (batch, channels, height, width).
        Returns
        -------
        upd_hidden : 5D hidden representation. (layer, batch, channels, height, width).
        '''
        
        for i in range(self.n_layers):
            input_ = x[i]
            cell = self.cells[i]
            if i != 0:
                input_ = torch.cat([input_, output_], dim=1)
            output_ = cell(input_)

        for i in range(self.n_output_layers):
            cell = self.cells[self.n_layers+i]
            output_ = cell(output_)

        # retain tensors in list to allow different hidden sizes
        return output_
    

class my_multi_GRU(nn.Module):
    def __init__(self, input_frames, output_frames, TyCatcher_input, TyCatcher_hidden, TyCatcher_n_layers, 
                encoder_input, encoder_downsample, encoder_gru, encoder_downsample_k, encoder_downsample_s, 
                encoder_downsample_p, encoder_gru_k, encoder_gru_s, encoder_gru_p, encoder_n_layers, 
                forecaster_upsample_cin, forecaster_upsample_cout, forecaster_upsample_k, forecaster_upsample_p, 
                forecaster_upsample_s, forecaster_n_layers, forecaster_output_cout=1, forecaster_output_k=1, 
                forecaster_output_s=1, forecaster_output_p=0, forecaster_n_output_layers=1, 
                batch_norm=False, device=None, value_dtype=None):
        super().__init__()
        self.device = device
        self.value_dtype = value_dtype
        self.input_frames = input_frames
        self.output_frames = output_frames

        self.tycatcher = TyCatcher(TyCatcher_input, TyCatcher_hidden, TyCatcher_n_layers, device, value_dtype)
        self.encoder1 = Encoder(encoder_input, encoder_downsample, encoder_gru, encoder_downsample_k, encoder_downsample_s, 
                                encoder_downsample_p, encoder_gru_k, encoder_gru_s, encoder_gru_p, encoder_n_layers,
                                batch_norm, device, value_dtype)
        self.encoder2 = Encoder(encoder_input, encoder_downsample, encoder_gru, encoder_downsample_k, encoder_downsample_s, 
                                encoder_downsample_p, encoder_gru_k, encoder_gru_s, encoder_gru_p, encoder_n_layers,
                                batch_norm, device, value_dtype)
        self.forecaster = Forecaster(forecaster_upsample_cin, forecaster_upsample_cout, forecaster_upsample_k, forecaster_upsample_p, forecaster_upsample_s, 
                                    forecaster_n_layers, forecaster_output_cout, forecaster_output_k, forecaster_output_s, forecaster_output_p, 
                                    forecaster_n_output_layers, batch_norm, device, value_dtype)

    def forward(self, inputs, ty_infos, radarmap):

        for i in range(self.input_frames):
            
            tmp_map = radarmap[:,i,:,:,:]
            if i == 0:
                prev_state = self.encoder1(input_, hidden=None)
            else:
                input_ = self.tycatcher(tmp_ty_info, tmp_map)
                prev_state = self.encoder1(input_, hidden=prev_state)

        outputs = []
        for i in range(self.output_frames):
            tmp_ty_info = ty_infos[:,i+self.input_frames,:]
            tmp_map = radar_map[:,-1,:,:,:]
            input_ = self.ty_catcher(tmp_ty_info, tmp_map)
            prev_state = self.encoder(input_)
            output_ = self.forecaster(prev_state)

            outputs.append(output_)
        outputs = torch.cat(outputs, dim=1)
        return outputs