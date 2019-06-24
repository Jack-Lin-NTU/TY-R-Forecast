import os 
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation 
from mpl_toolkits.basemap import Basemap
from .utils import createfolder

def plot_input(args, x, current_time, save=False):
    # set the size of basemap
    m = Basemap(projection='cyl', resolution='h', llcrnrlat=args.O_y[0], urcrnrlat=args.O_y[1], llcrnrlon=args.O_x[0], urcrnrlon=args.O_x[1])
    
    figures_folder = os.path.join(args.infers_folder)
    createfolder(os.path.join(figures_folder))

    X, Y = np.meshgrid(np.linspace(args.I_x[0],args.I_x[1],args.I_shape[0]), np.linspace(args.I_y[0],args.I_y[1],args.I_shape[1]))
    
    fig = plt.figure(figsize=(8*(args.input_frames/2),8), dpi=args.figure_dpi)

    for i in range(args.input_frames):
        print(i)
        ax = fig.add_subplot((args.input_frames//3), 3, i+1)
        data = x[i,0,:,:]
        _ = m.readshapefile(args.TW_map_file, name='Taiwan', linewidth=0.25, drawbounds=True, color='k', ax=ax)
        cs = m.contourf(x=X, y=Y, data=data, colors=args.RAD_cmap, levels=args.RAD_level, ax=ax)
        ax.set_xlabel(r'longtitude($^o$)',fontdict={'fontsize':10})
        ax.set_ylabel(r'latitude($^o$)',fontdict={'fontsize':10})
        _ = ax.set_xticks(ticks = np.linspace(args.O_x[0], args.O_x[1], 5))
        _ = ax.set_yticks(ticks = np.linspace(args.O_y[0], args.O_y[1], 5))
        ax.tick_params('both', labelsize=10)
        cbar = fig.colorbar(cs, ax=ax, shrink=0.8)
        cbar.ax.tick_params(labelsize=10)
    
    
    fig.suptitle('input_{}.png'.format(current_time))

    if save:
        fig.savefig(os.path.join(figures_folder, 'input_{}.png'.format(current_time)), dpi=args.figure_dpi, bbox_inches='tight')
    else:
        fig.tight_layout()
        plt.show()

def plot_target(args, x, current_time, save=False):
    # set the size of basemap
    m = Basemap(projection='cyl', resolution='h', llcrnrlat=args.O_y[0], urcrnrlat=args.O_y[1], llcrnrlon=args.O_x[0], urcrnrlon=args.O_x[1])
    
    figures_folder = os.path.join(args.infers_folder)
    createfolder(os.path.join(figures_folder))

    X, Y = np.meshgrid(np.linspace(args.I_x[0],args.I_x[1],args.I_shape[0]), np.linspace(args.I_y[0],args.I_y[1],args.I_shape[1]))
    
    fig = plt.figure(figsize=(8*(args.target_frames/2),8), dpi=args.figure_dpi)

    for i in range(args.target_frames):
        print(i)
        ax = fig.add_subplot((args.target_frames//3), 3, i+1)
        data = x[i,:,:]
        _ = m.readshapefile(args.TW_map_file, name='Taiwan', linewidth=0.25, drawbounds=True, color='k', ax=ax)
        cs = m.contourf(x=X, y=Y, data=data, colors=args.RAD_cmap, levels=args.RAD_level, ax=ax)
        ax.set_xlabel(r'longtitude($^o$)',fontdict={'fontsize':10})
        ax.set_ylabel(r'latitude($^o$)',fontdict={'fontsize':10})
        _ = ax.set_xticks(ticks = np.linspace(args.O_x[0], args.O_x[1], 5))
        _ = ax.set_yticks(ticks = np.linspace(args.O_y[0], args.O_y[1], 5))
        ax.tick_params('both', labelsize=10)
        cbar = fig.colorbar(cs, ax=ax, shrink=0.8)
        cbar.ax.tick_params(labelsize=10)
    
    
    fig.suptitle('target_{}.png'.format(current_time))

    if save:
        fig.savefig(os.path.join(figures_folder, 'target_{}.png'.format(current_time)), dpi=args.figure_dpi, bbox_inches='tight')
    else:
        fig.tight_layout()
        plt.show()
# def video(x):
