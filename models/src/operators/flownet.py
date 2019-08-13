import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def check_list(x, list_len):
    if type(x) != list:
        return [x] * list_len
    elif type(x) == list:
        if len(x) == 1:
            return x * list_len
        else:
            assert len(x) == list_len, 'Hyparameters are not matched len(c_hidden)'
            return x

class FlowNet(nn.Module):
    ''' A nn model for flow prediction. '''
    def __init__(self, c_in: int, c_hidden: list, kernel: list, stride: list, padding: list, batchnorm=False, initial_weight=True):
        super(FlowNet, self).__init__()
        self.layer_size = len(c_hidden)

        c_hidden = check_list(c_hidden, self.layer_size)
        kernel = check_list(kernel, self.layer_size)
        stride = check_list(stride, self.layer_size)
        padding = check_list(padding, self.layer_size)

        models = []
        for i in range(len(c_hidden)):
            if i == 0:
                models.append(nn.Conv2d(in_channels=c_in, out_channels=c_hidden[i], kernel_size=kernel[i], stride=stride[i], padding=padding[i]))
            else:
                models.append(nn.Conv2d(in_channels=c_hidden[i-1], out_channels=c_hidden[i], kernel_size=kernel[i], stride=stride[i], padding=padding[i]))
            
            if initial_weight is not None:
                nn.init.xavier_normal_(models[-1].weight)
                nn.init.constant_(models[-1].bias, 0)

            if batchnorm:
                models.append(nn.BatchNorm2d(c_hidden[i]))
            
        self.models = nn.Sequential(*models)
    
    def forward(self,x):
        return self.models(input=x)


