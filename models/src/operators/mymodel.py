import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .cnn2D import CNN2D_cell
from .convGRU import ConvGRUCell

class TyCatcher(nn.Module):
    def __init__(self, channel_input, channel_hidden, n_layers, device=None, value_dtype=None):
        super().__init__()
        self.device = device
        self.value_dtype = value_dtype

        if type(channel_hidden) != list:
            channel_hidden = [channel_hidden]*int(n_layers)
        else:
            assert len(channel_hidden) == n_layers, 'The length of `Channel_hidden` should be same as n_layers'

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

class myGRU(nn.Module):
    def __init__(self, input_frames, output_frames, TyCatcher_channel_input, TyCatcher_channel_hidden, TyCatcher_channel_n_layers, gru_channel_input, gru_channel_hidden, 
                gru_kernel, gru_stride, gru_padding, batch_norm=False, device=None, value_dtype=None):
        super().__init__()
        self.device = device
        self.value_dtype = value_dtype
        self.input_frames = input_frames
        self.output_frames = output_frames

        self.ty_catcher = TyCatcher(TyCatcher_channel_input, TyCatcher_channel_hidden, TyCatcher_channel_n_layers, device, value_dtype)
        self.model = ConvGRUCell(gru_channel_input, gru_channel_hidden, gru_kernel, gru_stride, gru_padding, batch_norm, device, value_dtype)

    def forward(self, ty_infos, radar_map):
        
        for i in range(self.input_frames):
            tmp_ty_info = ty_infos[:,i,:]
            tmp_map = radar_map[:,i,:,:,:]
            if i == 0:
                input_ = self.ty_catcher(tmp_ty_info, tmp_map)
                prev_state = self.model(input_, prev_state=None)
            else:
                input_ = self.ty_catcher(tmp_ty_info, tmp_map)
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