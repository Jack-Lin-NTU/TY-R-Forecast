import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .cnn2D import CNN2D_cell, DeCNN2D_cell
from .trajGRU import Encoder, Forecaster

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
        self.encoder = Encoder(channel_input=encoder_input_channel, channel_downsample=encoder_downsample_channels,
                                channel_rnn=encoder_rnn_channels, downsample_k=encoder_downsample_k, downsample_s=encoder_downsample_s, 
                                downsample_p=encoder_downsample_p, rnn_link_size=rnn_link_size, rnn_k=encoder_rnn_k, rnn_s=encoder_rnn_s, 
                                rnn_p=encoder_rnn_p, n_layers=encoder_n_layers, batch_norm=batch_norm, device=device, value_dtype=value_dtype)

        # forecasters
        self.forecaster = Forecaster(channel_input=forecaster_input_channel, channel_upsample=forecaster_upsample_channels, 
                                    channel_rnn=forecaster_rnn_channels, upsample_k=forecaster_upsample_k, upsample_s=forecaster_upsample_s, 
                                    upsample_p=forecaster_upsample_p, rnn_link_size=rnn_link_size, rnn_k=forecaster_rnn_k, 
                                    rnn_s=forecaster_rnn_s, rnn_p=forecaster_rnn_p, n_layers=forecaster_n_layers,
                                    channel_output=forecaster_output, output_k=forecaster_output_k, output_s=forecaster_output_s,
                                    output_p=forecaster_output_p, n_output_layers=forecaster_output_layers, batch_norm=batch_norm, 
                                    device=device, value_dtype=value_dtype)

    def forward(self, x):
        input_ = x
        if input_.size()[1] != self.n_encoders:
            assert input_.size()[1] == self.n_encoders, '`x` must have the same as n_encoders'

        forecast = []

        for i in range(self.n_encoders):
            if i == 0:
                hidden=None
            hidden = self.encoder(x = input_[:,i,:,:,:], hidden=hidden)

        hidden = hidden[::-1]

        for i in range(self.n_forecasters):
            hidden, output = self.forecaster(hidden=hidden)
            forecast.append(output)
            
        forecast = torch.cat(forecast, dim=1)
        
        return forecast
