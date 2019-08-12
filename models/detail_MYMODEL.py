## import useful tools
import os
import time
import datetime as dt
import numpy as np
import pandas as pd
pd.set_option('precision', 4)

import matplotlib.pyplot as plt
import matplotlib.animation as manimation
from mpl_toolkits.basemap import Basemap

## import torch modules
import torch
import torch.optim as optim

# import our model and dataloader
from src.utils.parser import get_args
from src.utils.utils import createfolder, remove_file, Adam16
from src.utils.loss import Criterion, LOSS
from src.runs.GRUs import get_dataloader, get_model, get_optimizer, test

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
    optimizer = get_optimizer(args=args, model=model)
    param_pt = os.path.join(args.params_folder, 'params_10.pt')
    checkpoint = torch.load(param_pt, map_location=args.device)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    # get trainloader and testloader
    trainloader, testloader = get_dataloader(args=args)

    model.eval()
    lev = [0.5, 2, 5, 10, 30]

    for idx, data in enumerate(testloader, 0):
        inputs, labels = data['inputs'].to(args.device, dtype=args.value_dtype), data['targets'].to(args.device, dtype=args.value_dtype)
        prediction_time = data['current_time'][0]

        if args.model.upper() == 'MYMODEL':
            ty_infos = data['ty_infos'].to(device=args.device, dtype=args.value_dtype)
            radar_map = data['radar_map'].to(device=args.device, dtype=args.value_dtype)
            outputs = model(encoder_inputs=inputs, ty_infos=ty_infos, radar_map=radar_map)
        else:
            outputs = model(inputs)                           # outputs.shape = [batch_size, target_frames, H, W]

        outputs = outputs.detach().to('cpu').numpy()
        labels = labels.to('cpu').numpy()
        if idx == 3:
            break
    labels = data['targets'].to('cpu').numpy()[0]
    labels[labels<1] = 0
    samples = model.samples(encoder_inputs=inputs, ty_infos=ty_infos, radar_map=radar_map).detach().to('cpu').numpy()[0]
    radar_map = radar_map.to('cpu').numpy()
    # breakpoint()
    fig = plt.figure(figsize=(16,8), dpi=args.figure_dpi)
    X, Y = np.meshgrid(np.linspace(args.I_x[0],args.I_x[1],args.I_shape[0]), np.linspace(args.I_y[0],args.I_y[1],args.I_shape[1])) 
    Xo, Yo = np.meshgrid(np.linspace(args.O_x[0],args.O_x[1],args.O_shape[0]), np.linspace(args.O_y[0],args.O_y[1],args.O_shape[1]))     
    
    m = Basemap(projection='cyl', resolution='h', llcrnrlat=args.O_y[0], urcrnrlat=args.O_y[1], llcrnrlon=args.O_x[0], urcrnrlon=args.O_x[1])

    createfolder(args.infers_folder)
    for i in range(18):
        images = os.path.join(args.infers_folder, args.model+str(i+1)+'.png')
        # 1
        ax = fig.add_subplot(1,2,1)
        _ = m.readshapefile(args.TW_map_file, name='Taiwan', linewidth=0.25, drawbounds=True, color='k', ax=ax)
        cs = m.contourf(x=X, y=Y, data=samples[-(i+1)], colors=args.RAD_cmap, levels=args.RAD_level, ax=ax)
        ax.set_xlabel(r'longtitude($^o$)',fontdict={'fontsize':10})
        ax.set_ylabel(r'latitude($^o$)',fontdict={'fontsize':10})
        _ = ax.set_xticks(ticks = np.linspace(args.I_x[0], args.I_x[1], 5))
        _ = ax.set_yticks(ticks = np.linspace(args.I_y[0], args.I_y[1], 5))
        ax.tick_params('both', labelsize=10)
        cbar = fig.colorbar(cs, ax=ax, shrink=0.8)
        cbar.ax.tick_params(labelsize=10)
        # 2
        ax = fig.add_subplot(1, 2, 2)
        _ = m.readshapefile(args.TW_map_file, name='Taiwan', linewidth=0.25, drawbounds=True, color='k', ax=ax)
        cs = m.contourf(x=X, y=Y, data=labels[i,:,:], colors=args.RAD_cmap, levels=args.RAD_level, ax=ax)
        ax.set_xlabel(r'longtitude($^o$)',fontdict={'fontsize':10})
        ax.set_ylabel(r'latitude($^o$)',fontdict={'fontsize':10})
        _ = ax.set_xticks(ticks = np.linspace(args.I_x[0], args.I_x[1], 5))
        _ = ax.set_yticks(ticks = np.linspace(args.I_y[0], args.I_y[1], 5))
        ax.tick_params('both', labelsize=10)
        cbar = fig.colorbar(cs, ax=ax, shrink=0.8)
        cbar.ax.tick_params(labelsize=10) 
        plt.title(i)
        fig.savefig(images, dpi=args.figure_dpi, bbox_inches='tight')
        fig.clf()