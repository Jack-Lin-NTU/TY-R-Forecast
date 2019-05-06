## import useful tools
import os
import time
import numpy as np
import pandas as pd
pd.set_option('precision', 4)

## import torch modules
import torch
import torch.optim as optim

# import our model and dataloader
from src.utils.argstools import args, createfolder, remove_file
from src.runs.GRUs_runs import get_dataloader, train, test

if __name__ == '__main__':
    pd.set_option('precision', 4)
    # set seed 
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # get trainloader and testloader
    trainloader, testloader = get_dataloader(args)
    # 
    # initilize model

    # set the factor of cnn channels
    c = args.channel_factor

    ## construct Traj GRU
    # initialize the parameters of the encoders and forecasters
    rnn_link_size = [13, 13, 9]

    encoder_input_channel = args.input_channels
    encoder_downsample_channels = [4*c,32*c,96*c]
    encoder_rnn_channels = [32*c,96*c,96*c]

    if args.I_shape[0] == args.F_shape[0]*3:
        encoder_downsample_k = [5,4,3]
        encoder_downsample_s = [3,2,2]
        encoder_downsample_p = [1,1,1]
    elif args.I_shape[0] == args.F_shape[0]:
        encoder_downsample_k = [7,4,3]
        encoder_downsample_s = [5,3,2]
        encoder_downsample_p = [1,1,1]

    encoder_rnn_k = [3,3,3]
    encoder_rnn_s = [1,1,1]
    encoder_rnn_p = [1,1,1]
    encoder_n_layers = 6

    forecaster_input_channel = 0
    forecaster_upsample_channels = [96*c,96*c,4*c]
    forecaster_rnn_channels = [96*c,96*c,32*c]

    forecaster_upsample_k = [3,4,7]
    forecaster_upsample_s = [2,3,5]
    forecaster_upsample_p = [1,1,1]

    forecaster_rnn_k = encoder_rnn_k
    forecaster_rnn_s = encoder_rnn_s
    forecaster_rnn_p = encoder_rnn_p
    forecaster_n_layers = encoder_n_layers

    forecaster_output_channels = 1
    forecaster_output_k = 3
    forecaster_output_s = 1
    forecaster_output_p = 1
    forecaster_output_layers = 1

    if args.model.upper() == 'TRAJGRU':
        from src.models.trajGRU_simple import Model
        print('Model: TRAJGRU')
        Net = Model(n_encoders=args.input_frames, n_forecasters=args.target_frames, rnn_link_size=rnn_link_size,
                encoder_input_channel=encoder_input_channel, encoder_downsample_channels=encoder_downsample_channels,
                encoder_rnn_channels=encoder_rnn_channels, encoder_downsample_k=encoder_downsample_k,
                encoder_downsample_s=encoder_downsample_s, encoder_downsample_p=encoder_downsample_p, 
                encoder_rnn_k=encoder_rnn_k,encoder_rnn_s=encoder_rnn_s, encoder_rnn_p=encoder_rnn_p, 
                encoder_n_layers=encoder_n_layers, forecaster_input_channel=forecaster_input_channel, 
                forecaster_upsample_channels=forecaster_upsample_channels, forecaster_rnn_channels=forecaster_rnn_channels,
                forecaster_upsample_k=forecaster_upsample_k, forecaster_upsample_s=forecaster_upsample_s, 
                forecaster_upsample_p=forecaster_upsample_p, forecaster_rnn_k=forecaster_rnn_k, forecaster_rnn_s=forecaster_rnn_s,
                forecaster_rnn_p=forecaster_rnn_p, forecaster_n_layers=forecaster_n_layers, forecaster_output=forecaster_output_channels, 
                forecaster_output_k=forecaster_output_k, forecaster_output_s=forecaster_output_s, 
                forecaster_output_p=forecaster_output_p, forecaster_output_layers=forecaster_output_layers, 
                batch_norm=args.batch_norm, device=args.device, value_dtype=args.value_dtype).to(args.device, dtype=args.value_dtype)
    elif args.model.upper() == 'CONVGRU':
        from src.models.convGRU_simple import Model
        print('Model: CONVGRU')
        Net = Model(n_encoders=args.input_frames, n_forecasters=args.target_frames,
                encoder_input_channel=encoder_input_channel, encoder_downsample_channels=encoder_downsample_channels,
                encoder_rnn_channels=encoder_rnn_channels, encoder_downsample_k=encoder_downsample_k,
                encoder_downsample_s=encoder_downsample_s, encoder_downsample_p=encoder_downsample_p, 
                encoder_rnn_k=encoder_rnn_k,encoder_rnn_s=encoder_rnn_s, encoder_rnn_p=encoder_rnn_p, 
                encoder_n_layers=encoder_n_layers, forecaster_input_channel=forecaster_input_channel, 
                forecaster_upsample_channels=forecaster_upsample_channels, forecaster_rnn_channels=forecaster_rnn_channels,
                forecaster_upsample_k=forecaster_upsample_k, forecaster_upsample_s=forecaster_upsample_s, 
                forecaster_upsample_p=forecaster_upsample_p, forecaster_rnn_k=forecaster_rnn_k, forecaster_rnn_s=forecaster_rnn_s,
                forecaster_rnn_p=forecaster_rnn_p, forecaster_n_layers=forecaster_n_layers, forecaster_output=forecaster_output_channels, 
                forecaster_output_k=forecaster_output_k, forecaster_output_s=forecaster_output_s, 
                forecaster_output_p=forecaster_output_p, forecaster_output_layers=forecaster_output_layers, 
                batch_norm=args.batch_norm, device=args.device, value_dtype=args.value_dtype).to(args.device, dtype=args.value_dtype)

    

    # train process
    time_s = time.time()

    args.result_folder += '_'+args.model.upper()
    args.params_folder += '_'+args.model.upper()

    size = '{}X{}'.format(args.I_shape[0], args.I_shape[1])

    if args.weather_list == []:
        args.result_folder = os.path.join(args.result_folder, size, 'RAD_no_weather')
        args.params_folder = os.path.join(args.params_folder, size, 'RAD_no_weather')
    else:
        args.result_folder = os.path.join(args.result_folder, size, 'RAD_weather')
        args.params_folder = os.path.join(args.params_folder, size, 'RAD_weather')

    args.result_folder = os.path.join(args.result_folder, 'wd{:.5f}_lr{:f}'.format(args.weight_decay, args.lr))
    args.params_folder = os.path.join(args.params_folder, 'wd{:.5f}_lr{:f}'.format(args.weight_decay, args.lr))

    if args.lr_scheduler and args.optimizer is not optim.Adam:
        args.result_folder += '_scheduler'
        args.params_folder += '_scheduler'
    
    if args.optimizer is optim.Adam:
        args.result_folder += '_Adam'
        args.params_folder += '_Adam'


    train(net=Net, trainloader=trainloader, testloader=testloader, loss_function=args.loss_function, args=args)

    time_e = time.time()
    t = time_e-time_s
    h = int((t)//3600)
    m = int((t-h*3600)//60)
    s = int(t-h*3600-m*60)

    print('The total computing time of this training process: {:d}:{:d}:{:d}'.format(h,m,s))
