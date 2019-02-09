import multiprocessing as mp
import matplotlib.pyplot as plt
import matplotlib.colors
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import FormatStrFormatter
from args_tools import *

def output_figure(fig_type, part):
    study_area = args.study_area
    # set lat and lon of inputs
    lat_l = args.I_lat_l
    lat_h = args.I_lat_h
    lon_l = args.I_lon_l
    lon_h = args.I_lon_h
    # Set path
    numpy_files_folder = args.numpy_files_folder
    figures_folder = os.path.join(args.figures_folder,study_area)

    TW_map_file = args.TW_map_file

    ty_list = pd.read_excel(args.ty_list)
    sta_list = pd.read_excel(args.sta_list, index_col="NO")

    man_num = 0
    for i in range(len(sta_list)):
        if type(sta_list.index[i]) == int:
            man_num += 1
        else:
            break

    # set specific color for radar, qpe, and qpf data
    levels_qp = [-1,0,10,20,35,50,80,120,160,200]
    c_qp = ('#FFFFFF','#D2D2FF','#AAAAFF','#8282FF','#6A6AFF','#4242FF','#1A1AFF','#000090','#000040','#000030')
    levels_rad = [-1,0,10,20,30,40,50,60,70]
    c_rad = ('#FFFFFF','#FFD8D8','#FFB8B8','#FF9090','#FF6060','#FF2020','#CC0000','#A00000')

    data_path = wrangled_files_folder+'/'+fig_type
    fig_path = wrangled_figs_folder+'/'+fig_type
    createfolder(fig_path)

    tmp=0
    if part == 1:
        start = 0
        end = int(len(sorted(os.listdir(data_path)))/3)
    elif part == 2:
        start = int(len(sorted(os.listdir(data_path)))/3)
        end = int(len(sorted(os.listdir(data_path)))/3*2)
    elif part == 3:
        start = int(len(sorted(os.listdir(data_path)))/3*2)
        end = len(sorted(os.listdir(data_path)))

    for j in sorted(os.listdir(data_path))[start:end]:
        if j[:-17] != tmp :
            tmp = j[:-17]
            print("|{:^18s}|".format(tmp))
        file_in = data_path+"/"+j
        data = np.load(file_in)
        x = np.linspace(lon_l,lon_h,len(data))
        y = np.linspace(lat_l,lat_h,len(data))
        plt.figure(figsize=(6,5))
        ax = plt.gca()
        m = Basemap(projection='cyl',resolution='h', llcrnrlat=lat_l, urcrnrlat=lat_h, llcrnrlon=lon_l, urcrnrlon=lon_h)
        m.readshapefile(TW_map_file, name='Taiwan', linewidth=0.25, drawbounds=True, color='gray')
        X, Y = np.meshgrid(x,y)

        if fig_type == "QPE" or fig_type == "QPF":
            cp = m.contourf(X,Y,data,levels_qp,colors=c_qp,alpha=0.95)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            cbar = plt.colorbar(cp,cax=cax)
            cbar.set_label('Rainfall (mm)',fontsize=10)
            # ax.plot(sta_list["Longitude"].iloc[man_num:].values, sta_list["Latitude"].iloc[man_num:].values, marker='.', mfc='#A00000', mec='#A00000', linestyle='None', markersize=1, label="Auto station")
            ax.plot(sta_list["Longitude"].iloc[:man_num].values, sta_list["Latitude"].iloc[:man_num].values, marker='^', mfc='#FF2020', mec='k', linestyle='None', markeredgewidth = 0.3, markersize=3, label="Man station")
            ax.legend(fontsize=7,loc='upper left')
        else:
            cp = m.contourf(X,Y,data,levels_rad,colors=c_rad,alpha=0.95)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            cbar = plt.colorbar(cp,cax=cax)
            cbar.set_label('Radar reflection (dbz)',fontsize=9)

        cbar.ax.tick_params(labelsize=8)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        title = j[:-17]
        ax.set_title(title,fontsize = 12)
        text_time = ""+j[-16:-12]+'/'+j[-12:-10]+"/"+j[-10:-8]+" "+j[-8:-6]+':'+j[-6:-4]
        ax.text(122.12,26.3,text_time,fontsize = 7)
        ax.set_xlabel(r"longitude$(^o)$",fontsize = 10)
        ax.set_ylabel(r"latitude$(^o)$",fontsize = 10)
        ax.set_xticks(np.linspace(args.lon_l,args.lon_h,6))
        ax.set_yticks(np.linspace(args.lat_l,args.lat_h,6))
        ax.tick_params(labelsize=8)
        figname = fig_path+'/'+j[:-4]+'.png'
        plt.savefig(figname,dpi=300,bbox_inches='tight')
        plt.close()

def multiprocess():
    tt = "*{:^18s}*".format(' Figure makers (multiprocessing) ')
    print("*" * len(tt))
    print(tt)
    print("*" * len(tt))

    # set multiprocessing
    p1 = mp.Process(target=output_figure,args=("RAD",1))
    p2 = mp.Process(target=output_figure,args=("QPE",1))
    p3 = mp.Process(target=output_figure,args=("QPF",1))
    p4 = mp.Process(target=output_figure,args=("RAD",2))
    p5 = mp.Process(target=output_figure,args=("QPE",2))
    p6 = mp.Process(target=output_figure,args=("QPF",2))
    p7 = mp.Process(target=output_figure,args=("RAD",3))
    p8 = mp.Process(target=output_figure,args=("QPE",3))
    p9 = mp.Process(target=output_figure,args=("QPF",3))
    p1.start()
    p2.start()
    p3.start()
    p4.start()
    p5.start()
    p6.start()
    p7.start()
    p8.start()
    p9.start()
    p1.join()
    p2.join()
    p3.join()
    p4.join()
    p5.join()
    p6.join()
    p7.join()
    p8.join()
    p9.join()

if __name__ == "__main__":
    multiprocess()
