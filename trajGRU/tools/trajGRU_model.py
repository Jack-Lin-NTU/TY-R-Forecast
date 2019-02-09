from .trajGRU_cell import *
from .cnn2D_model import *

class Encoder(nn.Module):
    def __init__(self, channel_input, channel_downsample, channel_rnn,
                downsample_k, downsample_s, downsample_p,
                rnn_link_size, rnn_k, rnn_s, rnn_p, n_layers,
                batch_norm=False):
        '''
        Generates a multi-layer convolutional GRU.
        Preserves spatial dimensions across cells, only altering depth.
        Parameters
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

        self.channel_input = channel_input

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

        cells = []
        for i in range(int(self.n_layers/2)):
            if i == 0:
                cell=CNN2D_cell(self.channel_input, self.channel_downsample[i], self.downsample_k[i],
                                self.downsample_s[i], self.downsample_p[i], batch_norm=batch_norm)
            else:
                cell=CNN2D_cell(self.channel_rnn[i-1], self.channel_downsample[i], self.downsample_k[i],
                                self.downsample_s[i], self.downsample_p[i], batch_norm=batch_norm)

            name = 'Downsample_' + str(i).zfill(2)
            setattr(self, name, cell)
            cells.append(getattr(self, name))

            cell = trajGRUCell(self.channel_downsample[i], self.channel_rnn[i], self.rnn_link_size[i],
                                self.rnn_k[i], self.rnn_s[i], self.rnn_p[i])
            name = 'trajGRUCell_' + str(i).zfill(2)
            setattr(self, name, cell)
            cells.append(getattr(self, name))

        self.cells = cells


    def forward(self, x, hidden=None):
        if not hidden:
            hidden = [None]*int(self.n_layers/2)

        input_ = x
        upd_hidden = []

        for i in range(self.n_layers):
            if i % 2 == 0:
                cell = self.cells[i]
                input_ = cell(input_)
            else:
                cell = self.cells[i]
                cell_hidden = hidden[int(i/2)]

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
                rnn_link_size, rnn_k, rnn_s, rnn_p, n_layers,
                channel_output=1, output_k=1, output_s = 1, output_p=0, n_output_layers=1,
                batch_norm=False):
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

        cells = []
        for i in range(int(self.n_layers/2)):
            if i == 0:
                cell = DetrajGRUCell(self.channel_input, self.channel_rnn[i], self.rnn_link_size[i],
                                    self.rnn_k[i], self.rnn_s[i], self.rnn_p[i])
            else:
                cell = DetrajGRUCell(self.channel_upsample[i-1], self.channel_rnn[i], self.rnn_link_size[i],
                                    self.rnn_k[i], self.rnn_s[i], self.rnn_p[i])

            name = 'DetrajGRUCelll_' + str(i).zfill(2)
            setattr(self, name, cell)
            cells.append(getattr(self, name))

            cell = DeCNN2D_cell(self.channel_rnn[i], self.channel_upsample[i], self.upsample_k[i], self.upsample_s[i],
                                self.upsample_p[i], batch_norm=batch_norm)
            name = 'Upsample_' + str(i).zfill(2)
            setattr(self, name, cell)
            cells.append(getattr(self, name))


        for i in range(self.n_output_layers):
            if i == 0:
                cell = nn.Conv2d(self.channel_upsample[-1], self.channel_output[i], self.output_k[i], self.output_s[i], self.output_p[i])
            else:
                cell = nn.Conv2d(self.channel_output[i-1], self.channel_output[i], self.output_k[i], self.output_s[i], self.output_p[i])
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

        for i in range(self.n_layers):
            if i % 2 == 0:
                cell = self.cells[i]
                cell_hidden = hidden[int(i/2)]
                # pass through layer

                upd_cell_hidden = cell(input=input_, prev_state=cell_hidden)

                upd_hidden.append(upd_cell_hidden)
                # update input_ to the last updated hidden layer for next pass
                input_ = upd_cell_hidden
            else:
                cell = self.cells[i]
                input_ = cell(input_)
        cell = self.cells[-1]
        output = cell(input_)

        # retain tensors in list to allow different hidden sizes
        return upd_hidden, output


class model(nn.Module):
    def __init__(self, n_encoders, n_decoders, rnn_link_size,
                encoder_input_channel, encoder_downsample_channels, encoder_rnn_channels,
                encoder_downsample_k, encoder_downsample_s, encoder_downsample_p,
                encoder_rnn_k, encoder_rnn_s, encoder_rnn_p, encoder_n_layers,
                decoder_input_channel, decoder_upsample_channels, decoder_rnn_channels,
                decoder_upsample_k, decoder_upsample_s, decoder_upsample_p,
                decoder_rnn_k, decoder_rnn_s, decoder_rnn_p, decoder_n_layers,
                decoder_output=1, decoder_output_k=1, decoder_output_s=1, decoder_output_p=0, decoder_output_layers=1,
                batch_norm=False):

        super().__init__()
        self.n_encoders = n_encoders
        self.n_decoders = n_decoders

        models = []
        for i in range(self.n_encoders):
            model = Encoder(channel_input=encoder_input_channel, channel_downsample=encoder_downsample_channels, channel_rnn=encoder_rnn_channels,
                            downsample_k=encoder_downsample_k, downsample_s=encoder_downsample_s, downsample_p=encoder_downsample_p,
                            rnn_link_size=rnn_link_size, rnn_k=encoder_rnn_k, rnn_s=encoder_rnn_s, rnn_p=encoder_rnn_p,
                            n_layers=encoder_n_layers, batch_norm=batch_norm)
            name = 'Encoder_' + str(i+1).zfill(2)
            setattr(self, name, model)
            models.append(getattr(self, name))

        for i in range(self.n_decoders):
            model = Forecaster(channel_input=decoder_input_channel, channel_upsample=decoder_upsample_channels,
                                channel_rnn=decoder_rnn_channels,
                                upsample_k=decoder_upsample_k, upsample_s=decoder_upsample_s, upsample_p=decoder_upsample_p,
                                rnn_link_size=rnn_link_size, rnn_k=decoder_rnn_k, rnn_s=decoder_rnn_s, rnn_p=decoder_rnn_p, n_layers=decoder_n_layers,
                                channel_output=decoder_output, output_k=decoder_output_k, output_s=decoder_output_s,
                                output_p=decoder_output_p, n_output_layers=decoder_output_layers, batch_norm=batch_norm)
            name = 'Forecaster_' + str(i+1).zfill(2)
            setattr(self, name, model)
            models.append(getattr(self, name))

        self.models = models

    def forward(self, x):
        if x.size()[1] != self.n_encoders:
            assert x.size()[1] == self.n_encoders, '`x` must have the same as n_encoders'

        forecast = []

        for i in range(self.n_encoders):
            if i == 0:
                hidden=None
            model = self.models[i]
            hidden = model(x = x[:,i,:,:,:], hidden=hidden)

        hidden = hidden[::-1]

        for i in range(self.n_encoders,self.n_encoders+self.n_decoders):

            model = self.models[i]
            hidden, output = model(hidden=hidden)
            forecast.append(output)
        forecast = torch.cat(forecast, dim=1)

        return forecast
