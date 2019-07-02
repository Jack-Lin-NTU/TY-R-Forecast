import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .cnn2D import CNN2D_cell, DeCNN2D_cell
from .convGRU import ConvGRUcell

class FlowNet(nn.Module):
    def __init__(self, channel_input, channel_hidden, k, s, p, batch_norm=False):
        super().__init__()


class Encoder(nn.Module):
    def __init__(self, channel_input, downsample_c, downsample_k, downsample_s, downsample_p, 
                gru_c, gru_k, gru_s, gru_p, batch_norm=False):
        super().__init__()

        if type(downsample_c) != list:
            downsample_c = [downsample_c]
        if type(downsample_k) != list:
            downsample_k = [downsample_k]*len(gru_c)
        if type(downsample_s) != list:
            downsample_s = [downsample_s]*len(gru_c)
        if type(downsample_p) != list:
            downsample_p = [downsample_p]*len(gru_c)
        if type(gru_c) != list:
            gru_c = [gru_c]
        if type(gru_k) != list:
            gru_k = [gru_k]*len(gru_c)
        if type(gru_s) != list:
            gru_s = [gru_s]*len(gru_c)
        if type(gru_p) != list:
            gru_p = [gru_p]*len(gru_c)
        
        self.n_cells = len(downsample_c)
        cells = []
        for i in range(self.n_cells):
            if i == 0:
                cell = CNN2D_cell(channel_input, downsample_c[i], downsample_k[i], downsample_s[i], downsample_p[i], batch_norm, initial_zeros=True)
            else:
                cell = CNN2D_cell(gru_c[i-1], downsample_c[i], downsample_k[i], downsample_s[i], downsample_p[i], batch_norm, initial_zeros=True)

            name = 'Downsample_' + str(i).zfill(2)
            setattr(self, name, cell)
            cells.append(getattr(self, name))

            cell = ConvGRUcell(downsample_c[i], gru_c[i], gru_k[i], gru_s[i], gru_p[i], batch_norm)
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
    def __init__(self, upsample_cin, upsample_cout, k, s, p, c_out, batch_norm=False):
        super().__init__()
        if type(upsample_cin) != list:
            upsample_cin = [upsample_cin]
        if type(upsample_cout) != list:
            upsample_cout = [upsample_cout]
        if type(k) != list:
            k = [k]*len(upsample_cout)
        if type(s) != list:
            s = [s]*len(upsample_cout)
        if type(p) != list:
            p = [p]*len(upsample_cout)

        self.n_layers = len(upsample_cout)

        layers = []
        for i in range(self.n_layers):
            if i == 0:
                layers.append(DeCNN2D_cell(upsample_cin[i], upsample_cout[i], k[i], s[i], p[i], batch_norm=batch_norm, initial_zeros=True))
            else:
                layers.append(DeCNN2D_cell(upsample_cin[i]+upsample_cout[i-1], upsample_cout[i], k[i], s[i], p[i], batch_norm=batch_norm, initial_zeros=True))
        self.layers = nn.Sequential(*layers)

    def grid_sample(self, x, flow):
        '''
        Function for sampling pixels based on given grid data.
        '''
        # get device and dtype
        device = self.layers[0].layer[0].weight.device
        dtype = self.layers[0].layer[0].weight.dtype
        input_ = x
        b, link_size, h, w = flow.shape
        y_, x_ = torch.meshgrid(torch.arange(h, device=device, dtype=dtype), torch.arange(w, device=device,dtype=dtype))
        y_, x_ = (y_*2/h-1).expand(b,-1,-1), (x_*2/w-1).expand(b,-1,-1)

        u, v = flow[:,0:int(link_size/2),:,:], flow[:,int(link_size/2):,:,:]
        samples = []
        for i in range(int(link_size/2)):
            new_x = x_ + u[:,i,:,:]
            new_y = y_ + v[:,i,:,:]
            grids = torch.stack([new_x, new_y], dim=3)
            samples.append(F.grid_sample(input_, grids, padding_mode='border'))

        return torch.cat(samples, dim=1)

    def forward(self, hidden_flow, map):
        for i, layer in enumerate(self.layers):
            if i == 0:
                stack_input = hidden_flow[i]
            else:
                stack_input = torch.cat([hidden_flow[i], output],dim=1)
            output = layer(stack_input)
        output = self.grid_sample(x=map, flow=output)
        output = torch.mean(output,dim=1).unsqueeze(1)
        return output

class Model(nn.Module):
    def __init__(self, input_frames, target_frames,
                encoder_cin, encoder_d_c, encoder_d_k, encoder_d_s, encoder_d_p,
                encoder_gru_c, encoder_gru_k, encoder_gru_s, encoder_gru_p,
                forecaster_u_cin, forecaster_u_cout, forecaster_u_k, forecaster_u_s, forecaster_u_p,
                c_out, batch_norm=False, target_RAD=False):
        super().__init__()
        self.encoder = Encoder(channel_input=encoder_cin, downsample_c=encoder_d_c, downsample_k=encoder_d_k, downsample_s=encoder_d_s, downsample_p=encoder_d_p,
                                gru_c=encoder_gru_c, gru_k=encoder_gru_k, gru_s=encoder_gru_s, gru_p=encoder_gru_p, batch_norm=batch_norm)
        self.forecaster = Forecaster(upsample_cin=forecaster_u_cin, upsample_cout=forecaster_u_cout, k=forecaster_u_k, s=forecaster_u_s, p=forecaster_u_p,
                                    c_out=c_out, batch_norm=False)
        self.input_frames = input_frames
        self.target_frames = target_frames
        self.target_RAD = target_RAD

    def forward(self, x, height):
        # x.shape = [b, input_frames, c, h, w]
        b,_,c,h,w = x.shape
        height = height.unsqueeze(1).expand(b,1,-1,-1)

        hidden_flow = [None]*3
        forecasts = []
        for i in range(self.input_frames-1):
            input_ = torch.cat([x[:,i:i+2,:,:,:].squeeze(2), height],dim=1)
            hidden_flow = self.encoder(x=input_, hidden=hidden_flow)
        
        output_ = self.forecaster(hidden_flow[::-1], x[:,-1,:,:,:])
        forecasts.append(output_)
        for i in range(self.target_frames-1):
            if i == 0:
                input_ = torch.cat([x[:,-1,:,:,:], forecasts[i], height], dim=1)
            else:
                input_ = torch.cat([forecasts[i-1], forecasts[i], height], dim=1)

            hidden_flow = self.encoder(x=input_, hidden=hidden_flow)
            output_ = self.forecaster(hidden_flow[::-1], forecasts[i])
            forecasts.append(output_)
        
        forecasts = torch.cat(forecasts, dim=1)
        if not self.target_RAD:
            forecast = ((10**(forecast/10))/200)**(5/8)

        return forecasts
