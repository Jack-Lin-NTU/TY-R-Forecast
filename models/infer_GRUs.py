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
from src.utils.visulize import plot_input,plot_target
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
    param_pt = os.path.join(args.params_folder, 'params_40.pt')
    checkpoint = torch.load(param_pt, map_location=args.device)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    # get trainloader and testloader
    trainloader, testloader = get_dataloader(args=args)

    model.eval()
    lev = [0.5, 2, 5, 10, 30]
    criterion = pd.DataFrame(np.zeros((2,len(lev))),index=['CSI', 'HSS'], columns=lev)
    for idx, data in enumerate(trainloader, 0):
        inputs, labels = data['inputs'].to(args.device, dtype=args.value_dtype), data['targets'].to(args.device, dtype=args.value_dtype)
        current_time = data['current_time'][0]

        if args.model.upper() == 'MYMODEL':
            ty_infos = data['ty_infos'].to(device=args.device, dtype=args.value_dtype)
            radar_map = data['radar_map'].to(device=args.device, dtype=args.value_dtype)
            outputs = model(encoder_inputs=inputs, ty_infos=ty_infos, radar_map=radar_map)
        else:
            outputs = model(inputs)                           # outputs.shape = [batch_size, target_frames, H, W]

        inputs = inputs.detach().to('cpu').numpy()[0]
        outputs = outputs.detach().to('cpu').numpy()[0]
        labels = labels.detach().to('cpu').numpy()[0]
        if idx == 50:
            break
    #     c = Criterion(outputs, labels)
    #     for threshold in lev:
    #         criterion[threshold] = criterion[threshold] + [c.csi(threshold)/len(trainloader), c.hss(threshold)/len(trainloader)]
    # criterion.to_csv('/home/jack/ssd/01_ty_research/criterion_{:s}.csv'.format(args.model.upper()))
    

    # inputs = testloader.dataset[0]['inputs']
    # current_time = trainloader.dataset[0]['current_time'][0]

    # plot_input(args,x=inputs,current_time=current_time)
    # targets = testloader.dataset[0]['targets']
    # current_time = trainloader.dataset[0]['current_time'][0]
    # plot_target(args,x=targets,current_time=current_time)
    outputs[outputs<1] = 0
    fig = plt.figure(figsize=(16,8), dpi=args.figure_dpi)
    X, Y = np.meshgrid(np.linspace(args.I_x[0],args.I_x[1],args.I_shape[0]),np.linspace(args.I_y[0],args.I_y[1],args.I_shape[1]))    
    
    m = Basemap(projection='cyl', resolution='h', llcrnrlat=args.I_y[0], urcrnrlat=args.I_y[1], llcrnrlon=args.I_x[0], urcrnrlon=args.I_x[1])

    times = outputs.shape[0]
    data_type = [args.model.upper(), 'Ground Truth']

    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='Movie Test', artist='Matplotlib', comment='Movie support!')
    writer = FFMpegWriter(fps=3, metadata=metadata)
    # breakpoint()
    video = os.path.join('/home/jack/ssd/Onedrive/01_IIS', args.model+'.mp4')

    current_time = dt.datetime.strptime(current_time,'%Y%m%d%H%M')
    with writer.saving(fig, video, 200):
        for i in range(times):
            data = [outputs[i], labels[i]]
            for idx in range(2):
                ax = fig.add_subplot(1, len(data_type), idx+1)
                _ = m.readshapefile(args.TW_map_file, name='Taiwan', linewidth=0.25, drawbounds=True, color='k', ax=ax)
                cs = m.contourf(x=X, y=Y, data=data[idx], colors=args.RAD_cmap, levels=args.RAD_level, ax=ax)
                ax.set_xlabel(r'longtitude($^o$)',fontdict={'fontsize':10})
                ax.set_ylabel(r'latitude($^o$)',fontdict={'fontsize':10})
                _ = ax.set_xticks(ticks = np.linspace(args.I_x[0], args.I_x[1], 5))
                _ = ax.set_yticks(ticks = np.linspace(args.I_y[0], args.I_y[1], 5))
                ax.tick_params('both', labelsize=10)
                cbar = fig.colorbar(cs, ax=ax, shrink=0.8)
                cbar.ax.tick_params(labelsize=10)
                # ax.legend(fontsize=10)
                ax.set_title(data_type[idx], fontsize=10)
                fig.suptitle(current_time+dt.timedelta(minutes=i*10))

            plt.tight_layout()
            writer.grab_frame()
            fig.clf()

    # # breakpoint()
    createfolder(args.infers_folder)
    for i in range(times):
        data = [outputs[i], labels[i]]
        images = os.path.join(args.infers_folder, args.model+str(i+1)+'.png')
        for idx in range(2):
            ax = fig.add_subplot(1, len(data_type), idx+1)
            _ = m.readshapefile(args.TW_map_file, name='Taiwan', linewidth=0.25, drawbounds=True, color='k', ax=ax)
            cs = m.contourf(x=X, y=Y, data=data[idx], colors=args.QPE_cmap, levels=args.QPE_level, ax=ax)
            ax.set_xlabel(r'longtitude($^o$)',fontdict={'fontsize':10})
            ax.set_ylabel(r'latitude($^o$)',fontdict={'fontsize':10})
            _ = ax.set_xticks(ticks = np.linspace(args.I_x[0], args.I_x[1], 5))
            _ = ax.set_yticks(ticks = np.linspace(args.I_y[0], args.I_y[1], 5))
            ax.tick_params('both', labelsize=10)
            cbar = fig.colorbar(cs, ax=ax, shrink=0.8)
            cbar.ax.tick_params(labelsize=10)
            # ax.legend(fontsize=10)
            ax.set_title(data_type[idx], fontsize=10)
            fig.suptitle(current_time+dt.timedelta(minutes=i*10))

        plt.tight_layout()
        fig.savefig(images, dpi=args.figure_dpi, bbox_inches='tight')
        fig.clf()