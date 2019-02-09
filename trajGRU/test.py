## import some useful tools
import os
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import datetime as dt

## import torch module
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, utils

# import our model and dataloader
from tools.args_tools import args, createfolder
from tools.datasetGRU import ToTensor, Normalize, TyDataset
from tools.loss_function import BMAE, BMSE
from tools.trajGRU_model import model

from torch.autograd import gradcheck


c = 2
# construct convGRU net
# initialize the parameters of the encoders and decoders


rnn_link_size = [13,13,9]
encoder_input_channel = 1
encoder_downsample_channels = [2*c,32*c,96*c]
encoder_rnn_channels = [32*c,96*c,96*c]



encoder_downsample_k = [3,4,4]
encoder_downsample_s = [1,2,2]
encoder_downsample_p = [1,1,1]


encoder_rnn_k = [3,3,3]
encoder_rnn_s = [1,1,1]
encoder_rnn_p = [1,1,1]
encoder_n_layers = 6

decoder_input_channel = 0
decoder_upsample_channels = [96*c,96*c,4*c]
decoder_rnn_channels = [96*c,96*c,32*c]

decoder_upsample_k = [4,4,3]
decoder_upsample_s = [2,2,1]
decoder_upsample_p = [1,1,1]

decoder_rnn_k = [3,3,3]
decoder_rnn_s = [1,1,1]
decoder_rnn_p = [1,1,1]
decoder_n_layers = 6

decoder_output = 1
decoder_output_k = 3
decoder_output_s = 1
decoder_output_p = 1
decoder_output_layers = 1

data = torch.randn(4,1,1,60,60).to("cuda", dtype=torch.float)
label = torch.randn(4,1,60,60).to("cuda", dtype=torch.float)
Net = model(n_encoders=1, n_decoders=1, rnn_link_size=rnn_link_size,
            encoder_input_channel=encoder_input_channel, encoder_downsample_channels=encoder_downsample_channels, encoder_rnn_channels=encoder_rnn_channels,
            encoder_downsample_k=encoder_downsample_k, encoder_downsample_s=encoder_downsample_s, encoder_downsample_p=encoder_downsample_p,
            encoder_rnn_k=encoder_rnn_k,encoder_rnn_s=encoder_rnn_s, encoder_rnn_p=encoder_rnn_p, encoder_n_layers=encoder_n_layers,
            decoder_input_channel=decoder_input_channel, decoder_upsample_channels=decoder_upsample_channels, decoder_rnn_channels=decoder_rnn_channels,
            decoder_upsample_k=decoder_upsample_k, decoder_upsample_s=decoder_upsample_s, decoder_upsample_p=decoder_upsample_p,
            decoder_rnn_k=decoder_rnn_k, decoder_rnn_s=decoder_rnn_s, decoder_rnn_p=decoder_rnn_p, decoder_n_layers=decoder_n_layers,
            decoder_output=1, decoder_output_k=decoder_output_k, decoder_output_s=decoder_output_s, decoder_output_p=decoder_output_p,
            decoder_output_layers=decoder_output_layers, batch_norm=False).to(args.device, dtype=torch.float)
# print(Net)
output = Net(data)
print(output.shape)
output = output.view(output.shape[0], -1)
label = label.view(label.shape[0], -1)
loss = BMSE(output, label)
optimizer = optim.Adam(Net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
optimizer.zero_grad()
loss.backward()
print('{:.4f} GB'.format(torch.cuda.max_memory_cached(0)/1024/1024/1024))
# for i in Net(data):
#     print(i.shape)
