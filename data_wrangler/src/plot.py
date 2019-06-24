import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.basemap import Basemap
import matplotlib.animation as manimation
from .utils.tools import createfolder

pd.set_option('precision',4)


def plot_data(args, x):
    m = Basemap(projection='cyl', resolution='h', llcrnrlat=args.O_y[0], urcrnrlat=args.O_y[1], llcrnrlon=args.O_x[0], urcrnrlon=args.O_x[1])

    X, Y = np.meshgrid(np.linspace(args.O_x[0],args.O_x[1],args.O_shape[0]), np.linspace(args.O_y[0],args.O_y[1],args.O_shape[1]))

    fig = plt.figure(figsize=(4,4), dpi=args.figure_dpi)
    
    ax = fig.add_subplot(1, 1, 1)
    
    _ = m.readshapefile(args.TW_map_file, name='Taiwan', linewidth=0.25, drawbounds=True, color='k', ax=ax)
    
    cs = m.contourf(x=X, y=Y, data=x, colors=args.RAD_cmap, levels=args.RAD_level, ax=ax)
    ax.set_xlabel(r'longtitude($^o$)',fontdict={'fontsize':10})
    ax.set_ylabel(r'latitude($^o$)',fontdict={'fontsize':10})
    _ = ax.set_xticks(ticks = np.linspace(args.O_x[0], args.O_x[1], 5))
    _ = ax.set_yticks(ticks = np.linspace(args.O_y[0], args.O_y[1], 5))
    ax.tick_params('both', labelsize=10)
    cbar = fig.colorbar(cs, ax=ax, shrink=0.8)
    cbar.ax.tick_params(labelsize=10)
    plt.show()


def plot(args, file_list, data_type):
    m = Basemap(projection='cyl', resolution='h', llcrnrlat=args.O_y[0], urcrnrlat=args.O_y[1], llcrnrlon=args.O_x[0], urcrnrlon=args.O_x[1])
    
    figures_folder = os.path.join(args.radar_figures_folder, 'Taiwan', 'RAD_and_QPE_denoise{}'.format(args.denoise))
    createfolder(os.path.join(figures_folder))

    X, Y = np.meshgrid(np.linspace(args.O_x[0],args.O_x[1],args.O_shape[0]), 
                    np.linspace(args.O_y[0],args.O_y[1],args.O_shape[1]))

    fig = plt.figure(figsize=(len(data_type)*8,8), dpi=args.figure_dpi)

    for i in range(len(file_list)):
        filename = file_list[i]
        trackname = file_list[i][:-17]
        tracktime = file_list[i][-16:-4]
        ty_track = pd.read_csv(os.path.join(args.ty_info_wrangled_data_folder, trackname+'.csv'))
        ty_track.Time = pd.to_datetime(ty_track.Time)
        ty_track.set_index('Time', inplace=True)
        ty_track = ty_track.reindex(columns=['Lat','Lon'])
        
        for idx in range(len(data_type)):
            ax = fig.add_subplot(1, len(data_type), idx+1)
            data = pd.read_pickle(os.path.join(args.radar_wrangled_data_folder, data_type[idx], filename), compression=args.compression)
            _ = m.readshapefile(args.TW_map_file, name='Taiwan', linewidth=0.25, drawbounds=True, color='k', ax=ax)
            cs = m.contourf(x=X, y=Y, data=data.to_numpy(), colors=args[data_type[idx]+'_cmap'], levels=args[data_type[idx]+'_level'], ax=ax)
            ax.scatter(x=ty_track.loc[tracktime].Lon, y=ty_track.loc[tracktime].Lat, marker='h', color='g', label='Ty Center')
            ax.plot(ty_track.Lon, ty_track.Lat, '--', color='gray', label='Ty Path')
            ax.set_xlabel(r'longtitude($^o$)',fontdict={'fontsize':10})
            ax.set_ylabel(r'latitude($^o$)',fontdict={'fontsize':10})
            _ = ax.set_xticks(ticks = np.linspace(args.O_x[0], args.O_x[1], 5))
            _ = ax.set_yticks(ticks = np.linspace(args.O_y[0], args.O_y[1], 5))
            ax.tick_params('both', labelsize=10)
            cbar = fig.colorbar(cs, ax=ax, shrink=0.8)
            cbar.ax.tick_params(labelsize=10)
            ax.legend(fontsize=10)
            ax.set_title(data_type[idx], fontsize=10)
            fig.suptitle(filename[:-4])
        fig.savefig(os.path.join(figures_folder, filename[:-4]+'.png'), dpi=args.figure_dpi, bbox_inches='tight')
        fig.clf()

def plot_p(args):
    return plot(*args)

def plot_parrallel(args):
    from multiprocessing import Pool
    data_type = ['QPE','RAD']
    file_list = sorted(os.listdir(os.path.join(args.radar_wrangled_data_folder, data_type[1])))
    chunks = [file_list[i::5] for i in range(5)]
    pool = Pool(processes=5)
    job_args = [(args, x, data_type) for x in chunks]
    # breakpoint()
    pool.map(plot_p, job_args)

def RAD_video(args, ty='2012.SAOLA'):
    file_list = [x for x in sorted(os.listdir(os.path.join(args.radar_wrangled_data_folder, 'RAD'))) if x[:10] == ty]

    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='Movie Test', artist='Matplotlib', comment='Movie support!')
    writer = FFMpegWriter(fps=3, metadata=metadata)
    fig = plt.figure(figsize=(8,8), dpi=args.figure_dpi)
    X, Y = np.meshgrid(np.linspace(args.O_x[0],args.O_x[1],args.O_shape[0]),np.linspace(args.O_y[0],args.O_y[1],args.O_shape[1]))    
    
    video = os.path.join(args.radar_figures_folder, ty+'.mp4')
    m = Basemap(projection='cyl', resolution='h', llcrnrlat=args.O_y[0], urcrnrlat=args.O_y[1], llcrnrlon=args.O_x[0], urcrnrlon=args.O_x[1])
    with writer.saving(fig, video, 200):
        for file in file_list:
            ax = fig.add_subplot(1,1,1)
            data = pd.read_pickle(os.path.join(args.radar_wrangled_data_folder, 'RAD', file), compression=args.compression).to_numpy()
            _ = m.readshapefile(args.TW_map_file, name='Taiwan', linewidth=0.25, drawbounds=True, color='k', ax=ax)
            cs = m.contourf(x=X, y=Y, data=data, colors=args.RAD_cmap, levels=args.RAD_level, ax=ax)
            ax.set_xlabel(r'longtitude($^o$)',fontdict={'fontsize':10})
            ax.set_ylabel(r'latitude($^o$)',fontdict={'fontsize':10})
            _ = ax.set_xticks(ticks = np.linspace(args.O_x[0], args.O_x[1], 5))
            _ = ax.set_yticks(ticks = np.linspace(args.O_y[0], args.O_y[1], 5))
            ax.tick_params('both', labelsize=10)
            cbar = fig.colorbar(cs, ax=ax, shrink=0.8)
            cbar.ax.tick_params(labelsize=10)
            # ax.legend(fontsize=10)
            ax.set_title('RAD\n{}'.format(file[-16:-4]), fontsize=12)

            # plt.tight_layout()
            writer.grab_frame()
            fig.clf()


if __name__ == '__main__':
    args = get_args()
    args.denoise = int(input("Please enter the threshold(dbz) of denoising: "))
    if args.denoise != 0:
        args.radar_wrangled_data_folder += '_denoise{}'.format(args.denoise)
    # plot(args)

    # parallel plotting
    plot_parrallel(args)
