import matplotlib.pyplot as plt
import matplotlib.colors
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1 import make_axes_locatable
#set ticks formats
from matplotlib.ticker import FormatStrFormatter
from args_tools import *


def output_continous_figures():
    # load typhoon list
    ty_list = pd.read_excel("../ty_list.xlsx")

    datafolder = "../03_wrangled_files_Taipei"
    # set output folder
    output_folder = "../05_continuous_fig_Taipei"
    createfolder(output_folder)

    data_list = []
    time = 12

    for i in ["RAD","QPE"]:
        path = os.path.join(datafolder,i)
        for j in range(int(time/2)):
            data_list.append(os.path.join(path,sorted(os.listdir(path))[j]))
    for i in ["RAD","QPE"]:
        path = os.path.join(datafolder,i)
        for j in range(int(time/2),time):
            data_list.append(os.path.join(path,sorted(os.listdir(path))[j]))

    plt.figure(figsize=(3*(time/2),10))
    count = 1

    for i in data_list:
        file_name = i[-16:-4]
        fig_type = i[-31:-28]
        if fig_type == "QPE":
            levels = [-5,0,10,20,35,50,80,120,160,200]
            c = ('#FFFFFF','#D2D2FF','#AAAAFF','#8282FF','#6A6AFF','#4242FF','#1A1AFF','#000090','#000040','#000030')
        else:
            levels = [-5,0,10,20,30,40,50,60,70]
            c = ('#FFFFFF','#FFD8D8','#FFB8B8','#FF9090','#FF6060','#FF2020','#CC0000','#A00000')

        ax = plt.subplot(4,int(time/2),count)
        file_path = i
        data = np.load(file_path)
        x = np.linspace(args.lon_l,args.lon_h, len(data))
        y = np.linspace(args.lat_l,args.lat_h,len(data))
        m = Basemap(projection='cyl',resolution='h', llcrnrlat=args.lat_l, urcrnrlat=args.lat_h, llcrnrlon=args.lon_l, urcrnrlon=args.lon_h)
        m.readshapefile(args.TW_map_file, name='Taiwan', linewidth=0.25, drawbounds=True, color='k')
        X, Y = np.meshgrid(x,y)

        if fig_type == "QPE":
            cp = m.contourf(X,Y,data,levels,colors=c,alpha=0.95)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = plt.colorbar(cp,cax=cax)
            cbar.set_label('Rainfall (mm)',fontsize=8)
        else:
            cp = m.contourf(X,Y,data,levels,colors=c,alpha=0.95)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar=plt.colorbar(cp,cax=cax)
            cbar.set_label('Radar reflection (dbz)',fontsize=8)

        h, _ = cp.legend_elements()
        ax.legend([h[0]], [file_name], loc=0, frameon=False, framealpha=0, fontsize=6)
        cbar.ax.xaxis.set_major_formatter(FormatStrFormatter('%02d'))
        cbar.ax.tick_params(labelsize=6)

        ax.set_title(fig_type,fontsize=10)
        ax.set_xlabel(r"longitude$(^o)$",fontsize = 8)
        ax.set_ylabel(r"latitude$(^o)$",fontsize = 8)

        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.set_xticks(np.linspace(args.lon_l,args.lon_h,5))
        ax.set_yticks(np.linspace(args.lat_l,args.lat_h,5))
        ax.tick_params(labelsize=5)
        print("Sub figure {:2d} is done!".format(count))
        count += 1

    output_path = os.path.join(output_folder,file_name)+".png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    output_continous_figures()
