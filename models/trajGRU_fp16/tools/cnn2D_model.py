import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN2D_cell(nn.Module):
    def __init__(self, channel_input, channel_hidden, kernel_size, stride=1, padding=1, batch_norm=False):
        super().__init__()

        self.channel_input = channel_input
        self.channel_hidden = channel_hidden
        self.padding = padding
        self.stride = stride

        layer_sublist = []
        layer_sublist.append(nn.Conv2d(channel_input, channel_hidden, kernel_size, stride=self.stride, padding=self.padding))
        if batch_norm:
            layer_sublist.append(nn.BatchNorm2d(channel_hidden))
        layer_sublist.append(nn.ReLU())

        nn.init.orthogonal_(layer_sublist[0].weight)
        nn.init.constant_(layer_sublist[0].bias, 0.)

        self.layer = nn.Sequential(*layer_sublist)

    def forward(self, input_):
        out = self.layer(input_)
        return out

class DeCNN2D_cell(nn.Module):
    def __init__(self, channel_input, channel_hidden, kernel_size, stride=1, padding=1, batch_norm=False):
        super().__init__()

        self.channel_input = channel_input
        self.channel_hidden = channel_hidden
        self.padding = padding
        self.stride = stride

        layer_sublist = []
        layer_sublist.append(nn.ConvTranspose2d(channel_input, channel_hidden, kernel_size, stride=self.stride, padding=self.padding))
        if batch_norm:
            layer_sublist.append(nn.BatchNorm2d(channel_hidden))
        layer_sublist.append(nn.ReLU())

        nn.init.orthogonal_(layer_sublist[0].weight)
        nn.init.constant_(layer_sublist[0].bias, 0.)

        self.layer = nn.Sequential(*layer_sublist)

    def forward(self, input_):
        out = self.layer(input_)
        return out

class Encoder(nn.Module):
    '''
    Generate a 2-D CNN encoder
    '''
    def __init__(self, channel_input, channel_hidden, kernel_size, stride, padding, n_layers, batch_norm=False):
        '''
        channel_input: integral. the channel size of input tensors.
        channel_hidden: integer or list. the channel size of hidden layers.
                if integral, the same hidden size is used for all layers.
        kernel_size: integer or list. the kernel size of each hidden layers.
                if integer, the same kernel size is used for all layers.
        stride: integer or list. the stride size of each hidden layers.
                if integer, the same stride size is used for all layers.
        padding: integer or list. the padding size of each hidden layers.
                if integer, the same padding size is used for all layers.
        n_layers: integral. the number of hidden layers (int)
        padding: boolean, the decision to do padding
        batch_norm = boolean, the decision to do batch normalization
        '''
        super().__init__()

        self.channel_input = channel_input

        if type(channel_hidden) != list:
            self.channel_hidden = [channel_hidden]*n_layers
        else:
            assert len(channel_hidden) == n_layers, '`channel_hidden` must have the same length as n_layers'
            self.channel_hidden = channel_hidden

        if type(kernel_size) != list:
            self.kernel_size = [kernel_size]*n_layers
        else:
            assert len(kernel_size) == n_layers, '`kernel_size` must have the same length as n_layers'
            self.kernel_size = kernel_size

        if type(stride) != list:
            self.stride = [stride]*n_layers
        else:
            assert len(stride) == n_layers, '`stride` must have the same length as n_layers'
            self.stride = stride

        if type(padding) != list:
            self.padding = [padding]*n_layers
        else:
            assert len(padding) == n_layers, '`padding` must have the same length as n_layers'
            self.padding = padding

        self.n_layers = n_layers

        # nn layers
        layers = []
        for layer_idx in range(self.n_layers):
            if layer_idx == 0:
                input_dim = self.channel_input
            else:
                input_dim = self.channel_hidden[layer_idx-1]

            layer = CNN2D_cell(input_dim, self.channel_hidden[layer_idx], self.kernel_size[layer_idx], stride=self.stride[layer_idx],
                                padding=self.padding[layer_idx], batch_norm=batch_norm)
            name = 'Conv_' + str(layer_idx).zfill(2)
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

        for layer_idx in range(self.n_layers):
            layer = self.layers[layer_idx]
            # pass through layers
            out_hidden = layer(input_)
            # update input_ to the last updated hidden layer for next pass
            input_ = out_hidden

        # retain tensors in list to allow different hidden sizes
        output = input_
        return output

class Decoder(nn.Module):
    '''
    Generate a 2-D CNN decoder.
    '''
    def __init__(self, channel_input, channel_hidden, kernel_size, stride, padding, n_layers, batch_norm=False):
        '''
        channel_input: integral. the channel size of input tensors.
        channel_hidden: integer or list. the channel size of hidden layers.
                if integral, the same hidden size is used for all layers.
        kernel_size: integer or list. the kernel size of each hidden layers.
                if integer, the same kernel size is used for all layers.
        stride: integer or list. the stride size of each hidden layers.
                if integer, the same stride size is used for all layers.
        padding: integer or list. the padding size of each hidden layers.
                if integer, the same padding size is used for all layers.
        n_layers: integral. the number of hidden layers (int)
        padding: boolean, the decision to do padding
        batch_norm = boolean, the decision to do batch normalization
        '''
        super().__init__()

        self.channel_input = channel_input

        if type(channel_hidden) != list:
            self.channel_hidden = [channel_hidden]*n_layers
        else:
            assert len(channel_hidden) == n_layers, '`channel_hidden` must have the same length as n_layers'
            self.channel_hidden = channel_hidden

        if type(kernel_size) != list:
            self.kernel_size = [kernel_size]*n_layers
        else:
            assert len(kernel_size) == n_layers, '`kernel_size` must have the same length as n_layers'
            self.kernel_size = kernel_size

        if type(stride) != list:
            self.stride = [stride]*n_layers
        else:
            assert len(stride) == n_layers, '`stride` must have the same length as n_layers'
            self.stride = stride

        if type(padding) != list:
            self.padding = [padding]*n_layers
        else:
            assert len(padding) == n_layers, '`padding` must have the same length as n_layers'
            self.padding = padding

        self.n_layers = n_layers

        # nn layers
        layers = []
        for layer_idx in range(self.n_layers):
            if layer_idx == 0:
                input_dim = self.channel_input
            else:
                input_dim = self.channel_hidden[layer_idx-1]

            layer = DeCNN2D_cell(input_dim, self.channel_hidden[layer_idx], self.kernel_size[layer_idx], stride=self.stride[layer_idx],
                                padding=self.padding[layer_idx], batch_norm=batch_norm)
            name = 'DeConv_' + str(layer_idx).zfill(2)
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

        for layer_idx in range(self.n_layers):
            layer = self.layers[layer_idx]
            # pass through layers
            out_hidden = layer(input_)
            # update input_ to the last updated hidden layer for next pass
            input_ = out_hidden

        # retain tensors in list to allow different hidden sizes
        output = input_
        return output


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
            assert len(n_hidden) == n_layers, '`kernel_size` must have the same length as n_layers'
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



class model(nn.Module):
    def __init__(self, encoder_input, encoder_hidden, encoder_kernel, encoder_n_layer, encoder_stride, encoder_padding,
                    decoder_input, decoder_hidden, decoder_kernel, decoder_n_layer, decoder_stride, decoder_padding,
                    fully_input=None, fully_hidden=None, fully_layers=None, batch_norm=False):
        super().__init__()
        self.encoder = Encoder(channel_input=encoder_input, channel_hidden=encoder_hidden, kernel_size=encoder_kernel,
                                n_layers=encoder_n_layer, stride=encoder_stride, padding=encoder_padding, batch_norm=batch_norm)
        self.decoder = Decoder(channel_input=decoder_input,channel_hidden=decoder_hidden,kernel_size=decoder_kernel,
                                n_layers=decoder_n_layer, stride=decoder_stride, padding=decoder_padding, batch_norm=batch_norm)
        self.fully = False
        if fully_input is not None:
            self.fully = True
            self.fc = Fully_Connect(n_input=fully_input,n_hidden=fully_hidden,n_layers=fully_layers)

    def forward(self, x):
        output = self.encoder(x)
        output = self.decoder(output)
        if self.fully:
            output = output.reshape(output.size(0), -1)
            output = self.fc(output)
        return output
