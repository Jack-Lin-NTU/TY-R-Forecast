import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN2D_cell(nn.Module):
    def __init__(self, channel_input, channel_output, kernel=3, stride=1, padding=1, batch_norm=False, negative_slope=0, initial_weight=None):
        super(CNN2D_cell, self).__init__()

        layer_sublist = []
        layer_sublist.append(nn.Conv2d(channel_input, channel_output, kernel, stride, padding))

        if batch_norm:
            layer_sublist.append(nn.BatchNorm2d(channel_output))
        # layer_sublist.append(nn.ReLU())

        if initial_weight is not None:
            nn.init.constant_(layer_sublist[0].weight, initial_weight)
            nn.init.zeros_(layer_sublist[0].bias)
        else:
            nn.init.kaiming_normal_(layer_sublist[0].weight, a=negative_slope, mode='fan_in', nonlinearity='leaky_relu')
            nn.init.kaiming_normal_(layer_sublist[0].bias, a=negative_slope, mode='fan_in', nonlinearity='leaky_relu')
        self.layer = nn.Sequential(*layer_sublist)
        
    def forward(self, x):
        out = self.layer(x)
        return out

class DeCNN2D_cell(nn.Module):
    def __init__(self, channel_input, channel_output, kernel=3, stride=1, padding=1, batch_norm=False, negative_slope=0, initial_weight=None):
        super().__init__()

        layer_sublist = []
        layer_sublist.append(nn.ConvTranspose2d(channel_input, channel_output, kernel, stride, padding))
        if batch_norm:
            layer_sublist.append(nn.BatchNorm2d(channel_output))
        # layer_sublist.append(nn.ReLU())
        
        if initial_weight is not None:
            nn.init.constant_(layer_sublist[0].weight, initial_weight)
            nn.init.zeros_(layer_sublist[0].bias)
        else:
            nn.init.kaiming_normal_(layer_sublist[0].weight, a=negative_slope, mode='fan_in', nonlinearity='leaky_relu')
            nn.init.kaiming_normal_(layer_sublist[0].bias, a=negative_slope, mode='fan_in', nonlinearity='leaky_relu')

        self.layer = nn.Sequential(*layer_sublist)

    def forward(self, x):
        out = self.layer(x)
        return out

class Encoder(nn.Module):
    '''
    Generate a 2-D CNN encoder
    '''
    def __init__(self, channel_input, channel_output, kernel, stride, padding, n_layers, batch_norm=False):
        '''
        channel_input: integral or list, the channel size of input size of each layer.
        channel_output: integer or list, the channel size of output size of each layer.
        kernel: integer or list, the kernel size of each layer.
        stride: integer or list, the stride size of each hidden layer.
        padding: integer or list, the padding size of each hidden layer.
        n_layers: integral. the number of layers.
        batch_norm: boolean, the decision to do batch normalization.
        '''
        super().__init__()
        if type(channel_input) != list:
            channel_input = [channel_input]*n_layers
        else:
            assert len(channel_input) == n_layers, '`channel_input` must have the same length as n_layers'

        if type(channel_output) != list:
            channel_output = [channel_output]*n_layers
        else:
            assert len(channel_output) == n_layers, '`channel_output` must have the same length as n_layers'

        if type(kernel) != list:
            kernel = [kernel]*n_layers
        else:
            assert len(kernel) == n_layers, '`kernel` must have the same length as n_layers'
           
        if type(stride) != list:
            stride = [stride]*n_layers
        else:
            assert len(stride) == n_layers, '`stride` must have the same length as n_layers'

        if type(padding) != list:
            padding = [padding]*n_layers
        else:
            assert len(padding) == n_layers, '`padding` must have the same length as n_layers'

        self.n_layers = n_layers

        # nn layers
        layers = []
        for i in range(self.n_layers):
            layer = CNN2D_cell(channel_input[i], channel_hidden[i], kernel[i], stride[i], padding[i], batch_norm)
            name = 'Conv2D_' + str(layer_idx).zfill(2)
            setattr(self, name, layer)
            layers.append(getattr(self, name))

        self.layers = layers

    def forward(self, x):
        '''
        Parameters
        ----------
        x : 4D input tensor. (batch, channels, height, width).
        '''
        input_ = x

        for i in range(self.n_layers):
            # pass through layers
            input_ = self.layers[i](input_)

        return input_

class Decoder(nn.Module):
    '''
    Generate a 2-D CNN decoder.
    '''
    def __init__(self, channel_input, channel_output, kernel, stride, padding, n_layers, batch_norm=False):
        '''
        channel_input: integral or list, the channel size of input size of each layer.
        channel_output: integer or list, the channel size of output size of each layer.
        kernel: integer or list, the kernel size of each layer.
        stride: integer or list, the stride size of each hidden layer.
        padding: integer or list, the padding size of each hidden layer.
        n_layers: integral. the number of layers.
        batch_norm: boolean, the decision to do batch normalization.
        '''
        super().__init__()
        if type(channel_input) != list:
            channel_input = [channel_input]*n_layers
        else:
            assert len(channel_input) == n_layers, '`channel_input` must have the same length as n_layers'

        if type(channel_output) != list:
            channel_output = [channel_output]*n_layers
        else:
            assert len(channel_output) == n_layers, '`channel_output` must have the same length as n_layers'

        if type(kernel) != list:
            kernel = [kernel]*n_layers
        else:
            assert len(kernel) == n_layers, '`kernel` must have the same length as n_layers'
           
        if type(stride) != list:
            stride = [stride]*n_layers
        else:
            assert len(stride) == n_layers, '`stride` must have the same length as n_layers'

        if type(padding) != list:
            padding = [padding]*n_layers
        else:
            assert len(padding) == n_layers, '`padding` must have the same length as n_layers'

        self.n_layers = n_layers

        # nn layers
        layers = []
        for i in range(self.n_layers):
            layer = CNN2D_cell(channel_input[i], channel_hidden[i], kernel[i], stride[i], padding[i], batch_norm)
            name = 'Conv2D_' + str(layer_idx).zfill(2)
            setattr(self, name, layer)
            layers.append(getattr(self, name))

        self.layers = layers

    def forward(self, x):
        '''
        Parameters
        ----------
        x : 4D input tensor. (batch, channels, height, width).
        '''
        input_ = x

        for i in range(self.n_layers):
            # pass through layers
            input_ = self.layers[i](input_)

        return input_


class Fully_Connect(nn.Module):
    '''
    Generate a fully connected layer
    '''
    def __init__(self, n_input, n_hidden, n_layers):
        super().__init__()
        self.n_input = n_input
        if type(n_hidden) != list:
            self.n_hidden = [n_hidden]*n_layers
        else:
            assert len(n_hidden) == n_layers, '`kernel` must have the same length as n_layers'
            self.n_hidden = n_hidden
        self.n_layers = n_layers

        layers = []
        for layer_idx in range(self.n_layers):
            if layer_idx == 0:
                input_dim = self.n_input
            else:
                input_dim = self.n_hidden[layer_idx-1]

            layer = nn.Linear(input_dim, self.n_hidden[layer_idx])
            name = 'Fc_' + str(layer_idx).zfill(2)
            setattr(self, name, layer)
            layers.append(getattr(self, name))

        self.layers = layers

    def forward(self, x):
        if x.dim() == 2:
            input_ = x
        else:
            input_ = x.reshape(x.size(0), -1)

        for layer_idx in range(self.n_layers):
            layer = self.layers[layer_idx]
            # pass through layers
            out_hidden = layer(input_)
            # update input_ to the last updated hidden layer for next pass
            input_ = out_hidden

        # retain tensors in list to allow different hidden sizes
        output = input_
        return output


class Model(nn.Module):
    def __init__(self, encoder_input, encoder_hidden, encoder_kernel, encoder_n_layer, encoder_stride, encoder_padding,
                    decoder_input, decoder_hidden, decoder_kernel, decoder_n_layer, decoder_stride, decoder_padding,
                    fully_input=None, fully_hidden=None, fully_layers=None, batch_norm=False):
        super().__init__()
        self.encoder = Encoder(channel_input=encoder_input, channel_hidden=encoder_hidden, kernel=encoder_kernel,
                                n_layers=encoder_n_layer, stride=encoder_stride, padding=encoder_padding, batch_norm=batch_norm)
        self.decoder = Decoder(channel_input=decoder_input,channel_hidden=decoder_hidden,kernel=decoder_kernel,
                                n_layers=decoder_n_layer, stride=decoder_stride, padding=decoder_padding, batch_norm=batch_norm)
        self.fully = False
        if fully_input is not None:
            self.fully = True
            self.fc = Fully_Connect(n_input=fully_input,n_hidden=fully_hidden,n_layers=fully_layers)

    def forward(self, x):
        input_ = x
        output = self.encoder(input_)
        output = self.decoder(output)
        if self.fully:
            output = output.reshape(output.size(0), -1)
            output = self.fc(output)
        return output
