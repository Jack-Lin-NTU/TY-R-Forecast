## import useful tools
import os
import time
import numpy as np
import pandas as pd
pd.set_option('precision', 4)

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

## import torch modules
import torch
import torch.optim as optim

# import our model and dataloader
from src.utils.parser import get_args
from src.utils.utils import createfolder, remove_file, Adam16
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
    param_pt = os.path.join(args.params_folder, 'params_100.pt')
    checkpoint = torch.load(param_pt, map_location=args.device)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    # get trainloader and testloader
    trainloader, testloader = get_dataloader(args=args)

    model.eval()

    for idx, data in enumerate(testloader, 0):
        if idx == 10:
            inputs, labels = data['inputs'].to(args.device, dtype=args.value_dtype), data['targets'].to(args.device, dtype=args.value_dtype)
            if args.model.upper() == 'MYMODEL':
                ty_infos = data['ty_infos'].to(device=args.device, dtype=args.value_dtype)
                radar_map = data['radar_map'].to(device=args.device, dtype=args.value_dtype)
                outputs = model(encoder_inputs=inputs, ty_infos=ty_infos, radar_map=radar_map)
            else:
                outputs = model(inputs)                           # outputs.shape = [batch_size, target_frames, H, W]
            break
    data = outputs[0,:,:,:].to('cpu').detach().numpy()
    labels = labels[0,:,:,:].to('cpu').detach().numpy()
    # breakpoint()
    fig, ax = plt.subplots(1,2,figsize=(16,8))
    X, Y = np.meshgrid(np.linspace(args.I_x[0],args.I_x[1],args.I_shape[0]),np.linspace(args.I_y[0],args.I_y[1],args.I_shape[1]))

    m = Basemap(projection='cyl', resolution='h', llcrnrlat=args.I_y[0], urcrnrlat=args.I_y[1], llcrnrlon=args.I_x[0], urcrnrlon=args.I_x[1], ax=ax[0])
    _ = m.readshapefile(args.TW_map_file, name='Taiwan', linewidth=0.25, drawbounds=True, color='k', ax=ax[0])
    m.contourf(X,Y,data[0], colors=args.QPE_cmap, levels=args.QPE_level, ax=ax[0])

    m = Basemap(projection='cyl', resolution='h', llcrnrlat=args.I_y[0], urcrnrlat=args.I_y[1], llcrnrlon=args.I_x[0], urcrnrlon=args.I_x[1], ax=ax[1])
    _ = m.readshapefile(args.TW_map_file, name='Taiwan', linewidth=0.25, drawbounds=True, color='k', ax=ax[1])
    m.contourf(X,Y,labels[0], colors=args.QPE_cmap, levels=args.QPE_level, ax=ax[1])

    # ax[1].contourf(X,Y,outputs[0,1,:,:],colors=args.QPE_cmap, levels=args.QPE_level)
    # ax[2].contourf(X,Y,outputs[0,2,:,:],colors=args.QPE_cmap, levels=args.QPE_level)
    # ax[3].contourf(X,Y,outputs[0,3,:,:],colors=args.QPE_cmap, levels=args.QPE_level)
    # ax[4].contourf(X,Y,outputs[0,4,:,:],colors=args.QPE_cmap, levels=args.QPE_level)
    fig.savefig('/home/jack/Desktop/test.png', dpi=120, bbox_inches='tight')
