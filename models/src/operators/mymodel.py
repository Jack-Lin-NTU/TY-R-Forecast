import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .cnn2D import CNN2D_cell, DeCNN2D_cell
from .convGRU import Encoder

class TyCatcher(nn.Module):
    def __init__(self, channel_input, channel_hidden, n_layers):
        super().__init__()

        if type(channel_hidden) != list:
            channel_hidden = [channel_hidden]*n_layers
        else:
            assert len(channel_hidden) == n_layers, 'The length of "Channel_hidden" should be same as n_layers'

        layers = []
        for i in range(n_layers):
            if i == 0:
                layer = nn.Linear(channel_input, channel_hidden[i])
            else:
                layer = nn.Linear(channel_hidden[i-1], channel_hidden[i])
            
            # nn.init.kaiming_normal_(layer.weight)
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

            name = 'Layer_' + str(i).zfill(2)
            setattr(self, name, layer)
            layers.append(getattr(self, name))

        self.n_layers = n_layers
        self.layers = layers
    
    def forward(self, ty_info, rader_map):
        # device and dtype
        device = self.layers[0].weight.device
        dtype = self.layers[0].weight.device

        b, c, _, _ = rader_map.shape
        # feed ty info to get theta
        output = ty_info
        for i in range(self.n_layers):
            layer = self.layers[i]
            output = layer(output)
        # breakpoint()
        output1, output2 = output.view(-1,2,3).chunk(2, dim=2)
        theta1 = torch.tensor([[1, 0],[0, 1]]).to(device=device, dtype=dtype).expand(b,2,2)
        theta2 = torch.zeros(b,2,1).to(device=device, dtype=dtype)
        theta = torch.cat([theta1+0.1*output1, theta2+output2], dim=2)
        size = torch.Size((b,c,400,400))
        flowfield = F.affine_grid(theta, size)
        sample = F.grid_sample(rader_map, flowfield)
        return sample

# class my_single_GRU(nn.Module):
#     def __init__(self, n_encoders, n_forecasters, TyCatcher_channel_input, TyCatcher_channel_hidden, TyCatcher_channel_n_layers, gru_channel_input, gru_channel_hidden, 
#                 gru_kernel, gru_stride, gru_padding, batch_norm=False, device=None, value_dtype=None):
#         super().__init__()
#         self.device = device
#         self.value_dtype = value_dtype
#         self.n_encoders = n_encoders
#         self.n_forecasters = n_forecasters

#         self.tycatcher = TyCatcher(TyCatcher_channel_input, TyCatcher_channel_hidden, TyCatcher_channel_n_layers, device, value_dtype)
#         self.model = ConvGRUCell(gru_channel_input, gru_channel_hidden, gru_kernel, gru_stride, gru_padding, batch_norm, device, value_dtype)

#     def forward(self, ty_infos, radar_map):
        
#         for i in range(self.n_encoders):
#             tmp_ty_info = ty_infos[:,i,:]
#             tmp_map = radar_map[:,i,:,:,:]
#             if i == 0:
#                 input_ = self.tycatcher(tmp_ty_info, tmp_map)
#                 prev_state = self.model(input_, prev_state=None)
#             else:
#                 input_ = self.tycatcher(tmp_ty_info, tmp_map)
#                 prev_state = self.model(input_)
        
#         outputs = []
#         for i in range(self.n_forecasters):
#             tmp_ty_info = ty_infos[:,i+self.n_encoders,:]
#             tmp_map = radar_map[:,-1,:,:,:]
#             input_ = self.ty_catcher(tmp_ty_info, tmp_map)
#             prev_state = self.model(input_)

#             outputs.append(prev_state)
#         outputs = torch.cat(outputs, dim=1)
#         return outputs

class Forecaster(nn.Module):
    def __init__(self, upsample_cin, upsample_cout, upsample_k, upsample_p, upsample_s, n_layers, 
                output_cout=1, output_k=1, output_s=1, output_p=0, n_output_layers=1,
                batch_norm=False):

        super().__init__()
        # channel size
        if type(upsample_cin) != list:
            upsample_cin = [upsample_cin]*n_layers
        assert len(upsample_cin) == n_layers, '"upsample_cin" must have the same length as n_layers'

        if type(upsample_cout) != list:
            upsample_cout = [upsample_cout]*n_layers
        assert len(upsample_cout) == n_layers, '"upsample_cout" must have the same length as n_layers'

        # kernel size
        if type(upsample_k) != list:
            upsample_k = [upsample_k]*n_layers
        assert len(upsample_k) == n_layers, '"upsample_k" must have the same length as n_layers'
        
       # stride size
        if type(upsample_s) != list:
            upsample_s = [upsample_s]*n_layers
        assert len(upsample_s) == n_layers, '"upsample_s" must have the same length as n_layers'

        # padding size
        if type(upsample_p) != list:
            upsample_p = [upsample_p]*n_layers
        assert len(upsample_p) == n_layers, '"upsample_p" must have the same length as n_layers'

        # output size
        if type(output_cout) != list:
            output_cout = [output_cout]*n_output_layers
        assert len(output_cout) == n_output_layers, '"output_cout" must have the same length as n_output_layers'

        if type(output_k) != list:
            output_k = [output_k]*n_output_layers
        assert len(output_k) == n_output_layers, '"output_k" must have the same length as n_output_layers'

        if type(output_p) != list:
            output_p = [output_p]*n_output_layers
        assert len(output_p) == n_output_layers, '"output_p" must have the same length as n_output_layers'

        if type(output_s) != list:
            output_s = [output_s]*n_output_layers
        assert len(output_s) == n_output_layers, '"output_s" must have the same length as n_output_layers'

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
    

class Model(nn.Module):
    def __init__(self, n_encoders, n_forecasters, TyCatcher_input, TyCatcher_hidden, TyCatcher_n_layers, 
                encoder_input, encoder_downsample, encoder_gru, encoder_downsample_k, encoder_downsample_s, 
                encoder_downsample_p, encoder_gru_k, encoder_gru_s, encoder_gru_p, encoder_n_layers, 
                forecaster_upsample_cin, forecaster_upsample_cout, forecaster_upsample_k, forecaster_upsample_p, 
                forecaster_upsample_s, forecaster_n_layers, forecaster_output_cout=1, forecaster_output_k=1, 
                forecaster_output_s=1, forecaster_output_p=0, forecaster_n_output_layers=1, 
                batch_norm=False):
        super().__init__()
        self.n_encoders = n_encoders
        self.n_forecasters = n_forecasters
        self.name = 'STN-CONVGRU'

        self.tycatcher = TyCatcher(TyCatcher_input, TyCatcher_hidden, TyCatcher_n_layers)
        self.encoder = Encoder(encoder_input, encoder_downsample, encoder_gru, encoder_downsample_k, encoder_downsample_s, 
                                encoder_downsample_p, encoder_gru_k, encoder_gru_s, encoder_gru_p, encoder_n_layers,
                                batch_norm)

        self.forecaster = Forecaster(forecaster_upsample_cin, forecaster_upsample_cout, forecaster_upsample_k, forecaster_upsample_p, forecaster_upsample_s, 
                                    forecaster_n_layers, forecaster_output_cout, forecaster_output_k, forecaster_output_s, forecaster_output_p, 
                                    forecaster_n_output_layers, batch_norm)

    def forward(self, inputs, ty_infos, radar_map):
        hidden = None
        for i in range(self.n_encoders):
            input_ = inputs[:,i,:,:,:]
            hidden = self.encoder(input_, hidden=hidden)
        
        forecast = []
        for i in range(self.n_forecasters):
            tmp_ty_info = ty_infos[:,i,:]
            input_ = self.tycatcher(tmp_ty_info, radar_map)
            hidden = self.encoder(input_, hidden=hidden)
            output_ = self.forecaster(hidden[::-1])
            forecast.append(output_)

        forecast = torch.cat(forecast, dim=1)
        forecast = ((10**(forecast/10))/200)**(5/8)
        return forecast


# class my_multi_GRU(nn.Module):
#     def __init__(self, n_encoders, n_forecasters, TyCatcher_input, TyCatcher_hidden, TyCatcher_n_layers, 
#                 encoder_input, encoder_downsample, encoder_gru, encoder_downsample_k, encoder_downsample_s, 
#                 encoder_downsample_p, encoder_gru_k, encoder_gru_s, encoder_gru_p, encoder_n_layers, 
#                 forecaster_upsample_cin, forecaster_upsample_cout, forecaster_upsample_k, forecaster_upsample_p, 
#                 forecaster_upsample_s, forecaster_n_layers, forecaster_output_cout=1, forecaster_output_k=1, 
#                 forecaster_output_s=1, forecaster_output_p=0, forecaster_n_output_layers=1, 
#                 batch_norm=False, device=None, value_dtype=None):
#         super().__init__()
#         self.device = device
#         self.value_dtype = value_dtype
#         self.n_encoders = n_encoders
#         self.n_forecasters = n_forecasters

#         self.tycatcher = TyCatcher(TyCatcher_input, TyCatcher_hidden, TyCatcher_n_layers, device, value_dtype)
#         encoders = []
#         for i in range(n_encoders+n_forecasters):
#             model = Encoder(encoder_input, encoder_downsample, encoder_gru, encoder_downsample_k, encoder_downsample_s, 
#                             encoder_downsample_p, encoder_gru_k, encoder_gru_s, encoder_gru_p, encoder_n_layers,
#                             batch_norm, device, value_dtype)
#             name = 'Encoder_' + str(i).zfill(2)
#             setattr(self, name, model)
#             encoders.append(getattr(self, name))

#         forecasters = []
#         for i in range(n_forecasters):
#             model = Forecaster(forecaster_upsample_cin, forecaster_upsample_cout, forecaster_upsample_k, forecaster_upsample_p, forecaster_upsample_s, 
#                                 forecaster_n_layers, forecaster_output_cout, forecaster_output_k, forecaster_output_s, forecaster_output_p, 
#                                 forecaster_n_output_layers, batch_norm, device, value_dtype)
#             name = 'Forecaster_' + str(i).zfill(2)
#             setattr(self, name, model)
#             forecasters.append(getattr(self, name))
        
#         self.encoders = encoders
#         self.forecasters = forecasters

#     def forward(self, encoder_inputs, ty_infos, radar_map):
#         for i in range(self.n_encoders):
#             input_ = encoder_inputs[:,i,:,:,:]
#             if i == 0:
#                 prev_state = self.encoders[i](input_, hidden=None)
#             else:
#                 prev_state = self.encoders[i](input_, hidden=prev_state)
        
#         outputs = []
#         for i in range(self.n_forecasters):
#             tmp_ty_info = ty_infos[:, i,:]
#             input_ = self.tycatcher(tmp_ty_info, radar_map)
#             prev_state = self.encoders[i+self.n_encoders](input_, hidden=prev_state)
#             output_ = self.forecasters[i](prev_state[::-1])

#             outputs.append(output_)
#         outputs = torch.cat(outputs, dim=1)
#         return outputs