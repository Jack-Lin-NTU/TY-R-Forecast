import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.basemap import Basemap
from src.utils.parser import get_args
from src.utils.tools import createfolder
from src.plot import plot_parrallel, RAD_video

pd.set_option('precision',4)


if __name__ == '__main__':
    args = get_args()
    args.denoise = int(input("Please enter the threshold(dbz) of denoising: "))
    args.smooth = int(input("Please enter number for smoothing (0:no, 1:yes): "))
    if args.denoise != 0:
        args.radar_wrangled_data_folder += '_denoise{}'.format(args.denoise)
    
    if args.smooth != 0:
        args.radar_wrangled_data_folder += '_smooth'
    # plot(args)

    # parallel plotting
#     plot_parrallel(args)
    
    RAD_video(args)