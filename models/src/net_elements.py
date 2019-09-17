from src.utils.utils import make_layers
from src.operators.convGRU import ConvGRUcell
from collections import OrderedDict

def get_elements(model):
    if 'CONVGRU' in model.upper():
        # build model
        encoder_elements = [
            [
                make_layers(OrderedDict({'conv1_leaky': [1, 8, 5, 3, 1]})),
                make_layers(OrderedDict({'conv2_leaky': [64, 192, 4, 2, 1]})),
                make_layers(OrderedDict({'conv3_leaky': [192, 192, 3, 2, 1]})),
            ],

            [
                ConvGRUcell(input_size=8, hidden_size=64),
                ConvGRUcell(input_size=192, hidden_size=192),
                ConvGRUcell(input_size=192, hidden_size=192),
            ]
        ]

        forecaster_elements = [
            [
                make_layers(OrderedDict({'deconv1_leaky': [192, 192, 3, 2, 1]})),
                make_layers(OrderedDict({'deconv2_leaky': [192, 64, 4, 2, 1]})),
                make_layers(OrderedDict({
                                        'deconv3_leaky': [64, 8, 5, 3, 1],
                                        'conv3_leaky': [8, 8, 3, 1, 1],
                                        'conv3': [8, 1, 1, 1, 0]
                                        })),
            ],

            [
                ConvGRUcell(input_size=192, hidden_size=192),
                ConvGRUcell(input_size=192, hidden_size=192),
                ConvGRUcell(input_size=64, hidden_size=64),
            ]
        ]


    return encoder_elements, forecaster_elements