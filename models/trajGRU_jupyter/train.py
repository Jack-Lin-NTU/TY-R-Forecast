import os
import time
import numpy as np
import pandas as pd
pd.set_option('precision', 4)

import torch
import torch.optim as optim

from tools.args_tools import args

from tools.trajGRU import Model
from tools.run import train, test, get_dataloader

# get trainloader and testloader
trainloader, testloader = get_dataloader(args)
# breakpoint()
# initilize model
inputs_channels = 1 + args.input_with_QPE*1 + len(args.weather_list) + args.input_with_grid*2

# set the factor of cnn channels
c = args.channel_factor

## construct Traj GRU
# initialize the parameters of the encoders and forecasters

rnn_link_size = [13, 8, 5]

encoder_input_channel = inputs_channels
encoder_downsample_channels = [9*c,32*c,96*c]
encoder_rnn_channels = [32*c,96*c,96*c]

forecaster_input_channel = 0
forecaster_upsample_channels = [96*c,96*c,4*c]
forecaster_rnn_channels = [96*c,96*c,32*c]

if args.I_shape[0] == args.F_shape[0]*3:
    encoder_downsample_k = [5,4,3]
    encoder_downsample_s = [3,2,2]
    encoder_downsample_p = [1,1,1]
elif args.I_shape[0] == args.F_shape[0]:
    encoder_downsample_k = [3,4,3]
    encoder_downsample_s = [1,2,2]
    encoder_downsample_p = [1,1,1]

encoder_rnn_k = [3,3,3]
encoder_rnn_s = [1,1,1]
encoder_rnn_p = [1,1,1]
encoder_n_layers = 6

forecaster_upsample_k = [3,4,3]
forecaster_upsample_s = [2,2,1]
forecaster_upsample_p = [1,1,1]

forecaster_rnn_k = [3,3,3]
forecaster_rnn_s = [1,1,1]
forecaster_rnn_p = [1,1,1]
forecaster_n_layers = 6

forecaster_output = 1
forecaster_output_k = 3
forecaster_output_s = 1
forecaster_output_p = 1
forecaster_output_layers = 1


Net = Model(args=args, n_encoders=args.input_frames, n_forecasters=args.target_frames, rnn_link_size=rnn_link_size,
            encoder_input_channel=encoder_input_channel, encoder_downsample_channels=encoder_downsample_channels,
            encoder_rnn_channels=encoder_rnn_channels, encoder_downsample_k=encoder_downsample_k,
            encoder_downsample_s=encoder_downsample_s, encoder_downsample_p=encoder_downsample_p, 
            encoder_rnn_k=encoder_rnn_k,encoder_rnn_s=encoder_rnn_s, encoder_rnn_p=encoder_rnn_p, 
            encoder_n_layers=encoder_n_layers, forecaster_input_channel=forecaster_input_channel, 
            forecaster_upsample_channels=forecaster_upsample_channels, forecaster_rnn_channels=forecaster_rnn_channels,
            forecaster_upsample_k=forecaster_upsample_k, forecaster_upsample_s=forecaster_upsample_s, 
            forecaster_upsample_p=forecaster_upsample_p, forecaster_rnn_k=forecaster_rnn_k, forecaster_rnn_s=forecaster_rnn_s,
            forecaster_rnn_p=forecaster_rnn_p, forecaster_n_layers=forecaster_n_layers, forecaster_output=forecaster_output, 
            forecaster_output_k=forecaster_output_k, forecaster_output_s=forecaster_output_s, 
            forecaster_output_p=forecaster_output_p, forecaster_output_layers=forecaster_output_layers, 
            batch_norm=args.batch_norm).to(args.device, dtype=args.value_dtype)
breakpoint()
# print(Net)
# train process
time_s = time.time()

size = '{}X{}'.format(args.I_shape[0], args.I_shape[1])

if args.weather_list == []:
    args.result_folder = os.path.join(args.result_folder, size, 'RAD_no_weather')
    args.params_folder = os.path.join(args.params_folder, size, 'RAD_no_weather')
else:
    args.result_folder = os.path.join(args.result_folder, size, 'RAD_weather')
    args.params_folder = os.path.join(args.params_folder, size, 'RAD_weather')

    
args.result_folder = os.path.join(args.result_folder, 'wd{:.5f}_lr{:f}'.format(args.weight_decay, args.lr))
args.params_folder = os.path.join(args.params_folder, 'wd{:.5f}_lr{:f}'.format(args.weight_decay, args.lr))

if args.lr_scheduler:
    args.result_folder += '_scheduler'
    args.params_folder += '_scheduler'

train(net=Net, trainloader=trainloader, testloader=testloader, args=args)

time_e = time.time()
t = time_e-time_s
h = int((t)//3600)
m = int((t-h*3600)//60)
s = int(t-h*3600-m*60)

print('The total computing time of this training process: {:d}:{:d}:{:d}'.format(h,m,s))