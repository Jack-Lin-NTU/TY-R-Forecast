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
import sys
sys.path.append("..")
from tools.args_tools import createfolder
from tools.dataset_GRU import ToTensor, Normalize, TyDataset
from tools.loss_function import BMAE, BMSE
from CNNGRU import *

#
# data = torch.randn(4,5,1,72,72).to("cuda")
#
# Net = model(n_encoders=5, n_decoders=20,
#         encoder_input=1, encoder_hidden=[2,3,4], encoder_kernel=[3,3,3], encoder_n_layers=3,
#         decoder_input=0, decoder_hidden=[4,3,2], decoder_output=1, decoder_kernel=[3,3,3], decoder_n_layers=3,
#         padding=True, batch_norm=False).to("cuda")
#
# # print(len(c(data)))
# # for i in c(data):
# #     print(i.size())
#
# input_frames = 5
# output_frames = 18
# # Normalize data
# mean = [12.834] * input_frames + [3.014] * output_frames
# std = [14.14] * input_frames + [6.773] * output_frames
# transfrom = transforms.Compose([ToTensor(),Normalize(mean=mean, std=std)])
# traindataset = TyDataset(ty_list_file="../../ty_list.xlsx",
#                     input_frames=input_frames,
#                     output_frames=output_frames,
#                     train=True,
#                     root_dir="../../01_TY_database/02_wrangled_data_Taipei/",
#                     transform = transfrom)
# testdataset = TyDataset(ty_list_file="../../ty_list.xlsx",
#                     input_frames=input_frames,
#                     output_frames=output_frames,
#                     train=False,
#                     root_dir="../../01_TY_database/02_wrangled_data_Taipei/",
#                     transform = transfrom)
#
# # set train and test dataloader
# params = {"batch_size":10, "shuffle":True, "num_workers":1}
# trainloader = DataLoader(traindataset, **params)
# testloader = DataLoader(testdataset, **params)
# n = 0
# for idx, data in enumerate(trainloader):
#     rad = data["RAD"].to("cuda",dtype=torch.float)
#     label = data["QPE"].to("cuda",dtype=torch.float)
#     n += label.size(0) * label.size(1)
#     # print(idx, results.size())
#     print(label.size(0), label.size(1))
# print(n)

data = torch.randn(4,10,1,180,180).to("cuda", dtype=torch.float)
c = 2
encoder_input = 1
encoder_downsample = [2*c,32*c,96*c]
encoder_crnn = [32*c,96*c,96*c]
encoder_kernel_downsample = [5,4,4]
encoder_kernel_crnn = [3,3,3]
encoder_stride_downsample = [3,2,2]
encoder_stride_crnn = [1,1,1]
encoder_padding_downsample = [1,1,1]
encoder_padding_crnn = [1,1,1]
encoder_n_layers = 6

decoder_input=0
decoder_upsample = [96*c,96*c,4*c]
decoder_crnn = [96*c,96*c,32*c]
decoder_kernel_upsample = [4,4,5]
decoder_kernel_crnn = [3,3,3]
decoder_stride_upsample = [2,2,3]
decoder_stride_crnn = [1,1,1]
decoder_padding_upsample = [1,1,1]
decoder_padding_crnn = [1,1,1]
decoder_n_layers = 6
decoder_output = 1
decoder_output_kernel = 5
decoder_output_stride = 3
decoder_output_padding = 1
decoder_output_layers = 1
# net1 = ConvGRU(encoder_input, encoder_downsample, encoder_crnn, encoder_kernel_downsample, encoder_kernel_crnn,
#                 encoder_stride_downsample, encoder_stride_crnn, encoder_padding_downsample, encoder_padding_crnn, encoder_n_layers).to("cuda", dtype=torch.float)
# net2 = DeconvGRU(decoder_input, decoder_upsample, decoder_crnn, decoder_kernel_upsample, decoder_kernel_crnn,
#                 decoder_stride_upsample, decoder_stride_crnn, decoder_padding_upsample, decoder_padding_crnn, decoder_n_layers).to("cuda", dtype=torch.float)

Net = model(n_encoders=10, n_decoders=18,
            encoder_input=encoder_input, encoder_downsample=encoder_downsample, encoder_crnn=encoder_crnn,
            encoder_kernel_downsample=encoder_kernel_downsample, encoder_kernel_crnn=encoder_kernel_crnn,
            encoder_stride_downsample=encoder_stride_downsample, encoder_stride_crnn=encoder_stride_crnn,

            encoder_padding_downsample=encoder_padding_downsample, encoder_padding_crnn=encoder_padding_crnn, encoder_n_layers=encoder_n_layers,
            decoder_input=decoder_input, decoder_upsample=decoder_upsample, decoder_crnn=decoder_crnn,
            decoder_kernel_upsample=decoder_kernel_upsample, decoder_kernel_crnn=decoder_kernel_crnn,
            decoder_stride_upsample=decoder_stride_upsample, decoder_stride_crnn=decoder_stride_crnn,
            decoder_padding_upsample=decoder_padding_upsample, decoder_padding_crnn=decoder_padding_crnn,
            decoder_n_layers=decoder_n_layers, decoder_output=1, decoder_output_kernel= decoder_output_kernel,
            decoder_output_stride=decoder_output_stride, decoder_output_padding=decoder_output_padding,
            decoder_output_layers=decoder_output_layers, batch_norm=False).to("cuda", dtype=torch.float)
# for i in net1(data):
#     print(i.size())
print(Net(data).size())
