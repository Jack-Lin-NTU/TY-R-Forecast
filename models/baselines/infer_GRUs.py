## import useful tools
import os
import time
import numpy as np
import pandas as pd
pd.set_option('precision', 4)

import matplotlib.pyplot as plt

## import torch modules
import torch
import torch.optim as optim

# import our model and dataloader
from src.utils.parser import get_args
from src.utils.utils import createfolder, remove_file, Adam16
from src.runs.GRUs import get_dataloader, get_model, test

if __name__ == "__main__":
    args = get_args()
    pd.set_option('precision', 4)
    # set seed 
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Get the model
    model = get_model(args=args)

    # set optimizer
    if args.optimizer == 'Adam16':
        optimizer = Adam16(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, device=args.device)
    else:
        optimizer = getattr(optim, args.optimizer)
        if args.optimizer == 'Adam16':
            optimizer = optimizer(model.parameters(), lr=args.lr, eps=1e-07, weight_decay=args.weight_decay)
        elif args.optimizer == 'SGD':
            optimizer = optimizer(model.parameters(), lr=args.lr, momentum=0.6, weight_decay=args.weight_decay)
        else:
            optimizer = optimizer(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    args.params_folder += '_'+args.model.upper()
    size = '{}X{}'.format(args.I_shape[0], args.I_shape[1])
    if args.weather_list == []:
        args.params_folder = os.path.join(args.params_folder, size, 'RAD_no_weather')
    else:
        args.params_folder = os.path.join(args.params_folder, size, 'RAD_weather')
    args.params_folder = os.path.join(args.params_folder, 'wd{:.5f}_lr{:f}'.format(args.weight_decay, args.lr))
    if args.lr_scheduler and args.optimizer != 'Adam':
        args.params_folder += '_scheduler'
    args.params_folder += '_'+args.optimizer

    param_pt = os.path.join(args.params_folder, 'params.pt')
    checkpoint = torch.load(param_pt)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    # get trainloader and testloader
    trainloader, testloader = get_dataloader(args=args)

    model.eval()

    for idx, data in enumerate(trainloader, 0):
        if idx == 10:
            inputs, labels = data['inputs'].to(args.device, dtype=args.value_dtype), data['targets'].to(args.device, dtype=args.value_dtype)
            outputs = model(inputs).to('cpu').detach().numpy()
            # outputs = outputs*(args.max_values['QPE'] - args.min_values['QPE'])+args.min_values['QPE']
            break
    breakpoint()
    fig, ax = plt.subplots(1,5,figsize=(40,8))
    X, Y = np.meshgrid(np.linspace(args.I_x[0],args.I_x[1],args.I_shape[0]),np.linspace(args.I_y[0],args.I_y[1],args.I_shape[1]))
    ax[0].contourf(X,Y,outputs[0,0,:,:],colors=args.QPE_cmap, levels=args.QPE_level)
    ax[1].contourf(X,Y,outputs[0,1,:,:],colors=args.QPE_cmap, levels=args.QPE_level)
    ax[2].contourf(X,Y,outputs[0,2,:,:],colors=args.QPE_cmap, levels=args.QPE_level)
    ax[3].contourf(X,Y,outputs[0,3,:,:],colors=args.QPE_cmap, levels=args.QPE_level)
    ax[4].contourf(X,Y,outputs[0,4,:,:],colors=args.QPE_cmap, levels=args.QPE_level)
    fig.savefig('/home/jack/Desktop/test.png')
