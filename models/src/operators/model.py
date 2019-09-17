from torch import nn
import torch.nn.functional as F
import torch
from src.utils.utils import make_layers

class activation():
    def __init__(self, act_type, negative_slope=0.2, inplace=True):
        super().__init__()
        self._act_type = act_type
        self.negative_slope = negative_slope
        self.inplace = inplace

    def __call__(self, input_):
        if self._act_type == 'leaky':
            return F.leaky_relu(input_, negative_slope=self.negative_slope, inplace=self.inplace)
        elif self._act_type == 'relu':
            return F.relu(input_, inplace=self.inplace)
        elif self._act_type == 'sigmoid':
            return torch.sigmoid(input_)
        else:
            raise NotImplementedError

class EF(nn.Module):
    def __init__(self, encoder, forecaster):
        super().__init__()
        self.encoder = encoder
        self.forecaster = forecaster

    def forward(self, input_):
        state = self.encoder(input_)
        output = self.forecaster(state)
        return output

def get_model(args):
    from src.net_elements import get_elements
    batch_size = args.batch_size
    IN_LEN = args.I_nframes
    PRED_LEN = args.F_nframes
    encoder_elements, forecaster_elements = get_elements(args.model)
    encoder = Encoder(subnets=encoder_elements[0], rnns=encoder_elements[1])
    forecaster = Forecaster(subnets=forecaster_elements[0], rnns=forecaster_elements[1], seq_len=PRED_LEN)
    return EF(encoder, forecaster)