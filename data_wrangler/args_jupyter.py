# import argparse
import easydict
import numpy as np
import pandas as pd
import math
import datetime as dt
import os


def createfolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

def make_path(path, workfolder=None):
    if path[0] == '~':
        p = os.path.expanduser(path)
    else:
        p = path
    
    if workfolder is not None and workfolder[0] == '~':
        workfolder = os.path.expanduser(workfolder)
    else:
        workfolder = workfolder
    
    if workfolder is not None and not os.path.isabs(p):
        return os.path.join(os.path.expanduser(workfolder), p)
    else:
        return p

args = easydict.EasyDict({
    'study_area': 'Taipei',
    'radar_folder': os.path.expanduser('~/Onedrive/01_IIS/04_TY_research/01_Radar_data'),
    'I_lat_l': 23.9125,
    'I_lat_h': 26.15,
    'I_lon_l': 120.4,
    'I_lon_h': 122.6375,
    'F_lat_l': 24.6625,
    'F_lat_h': 25.4,
    'F_lon_l': 121.15,
    'F_lon_h': 121.8875,
    'origin_lat_l': 20,
    'origin_lat_h': 27,
    'origin_lon_l': 118,
    'origin_lon_h': 123.5,
    'res_degree': 0.0125
})

args.ty_list = make_path('ty_list.xlsx', args.radar_folder)
args.sta_list = make_path('sta_list_all.xlsx', args.radar_folder)
args.TW_map_file = make_path('TW_shapefile/gadm36_TWN_2', args.radar_folder)
args.fortran_code_folder = 'fortran_codes/'
args.original_files_folder = os.path.expanduser('~/ssd/research/origianal_radar_data_2012_2018')
args.compressed_files_folder = make_path('01_compressed_files', args.radar_folder)
args.files_folder = make_path('02_pandas_files', args.radar_folder)
args.figures_folder = make_path('03_figures', args.radar_folder)

args.I_shape = (math.ceil((args.I_lon_h-args.I_lon_l)/args.res_degree)+1,math.ceil((args.I_lat_h-args.I_lat_l)/args.res_degree)+1)
args.F_shape = (math.ceil((args.F_lon_h-args.F_lon_l)/args.res_degree)+1,math.ceil((args.F_lat_h-args.F_lat_l)/args.res_degree)+1)
args.origin_size = (math.ceil((args.origin_lat_h-args.origin_lat_l)/args.res_degree)+1, math.ceil((args.origin_lon_h-args.origin_lon_l)/args.res_degree)+1)

args.I_x_left = int((args.I_lon_l-args.origin_lon_l)/args.res_degree + 1)
args.I_x_right = int(args.I_x_left + (args.I_lon_h-args.I_lon_l)/args.res_degree + 1)
args.I_y_low = int((args.I_lat_l-args.origin_lat_l)/args.res_degree + 1)
args.I_y_high = int(args.I_y_low + (args.I_lat_h-args.I_lat_l)/args.res_degree + 1)
