# import argparse
import os
import math
import easydict
import numpy as np
import pandas as pd
import datetime as dt

def checkpath(path):
    if os.path.exists(path):
        print('Path exists!')
    else:
        print('Not exists!')
        
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
    'working_folder': os.path.expanduser('~/ssd/01_ty_research'),
    'I_y': [23.9125, 26.15],
    'I_x': [120.4, 122.6375],
    'F_y': [24.6625, 25.4],
    'F_x': [121.15, 121.8875],
    'O_y': [20, 27],
    'O_x': [118,123.5],
    'res_degree': 0.0125,
    'compression': 'bz2',
    'figure_dpi': 150,
    'RAD_cmap': ['#FFFFFF','#FFD8D8','#FFB8B8','#FF9090','#FF6060','#FF2020','#CC0000','#A00000'],
    'RAD_level': [-5, 0, 10, 20, 30, 40, 50, 60, 70],
    'QPE_cmap': ['#FFFFFF','#D2D2FF','#AAAAFF','#8282FF','#6A6AFF','#4242FF','#1A1AFF','#000090','#000040','#000030'],
    'QPE_level': [-5, 0, 10, 20, 35, 50, 80, 120, 160, 200],
    'QPF_cmap': ['#FFFFFF','#D2D2FF','#AAAAFF','#8282FF','#6A6AFF','#4242FF','#1A1AFF','#000090','#000040','#000030'],
    'QPF_level': [-5, 0, 10, 20, 35, 50, 80, 120, 160, 200]
})

args.ty_list = make_path('ty_list.xlsx', args.working_folder)
args.sta_list = make_path('sta_list_all.xlsx', args.working_folder)
args.TW_map_file = make_path(os.path.join('07_gis_data','03_TW_shapefile','gadm36_TWN_2'), args.working_folder)
args.fortran_code_folder = make_path('08_fortran_codes', args.working_folder)

# To set the path of the radar folders
args.radar_folder = make_path('01_radar_data', args.working_folder)
args.radar_original_data_folder = make_path(os.path.join('00_original_data', 'radar_data_2012_2018'), args.working_folder)
args.radar_compressed_data_folder = make_path('01_compressed_files', args.radar_folder)
args.radar_wrangled_data_folder = make_path('02_wrangled_files', args.radar_folder)
args.radar_figures_folder = make_path('03_figures', args.radar_folder)

# To set the path of meteorology data folders
args.meteorology_folder = make_path('02_meteorological_data', args.working_folder)
args.meteorology_wrangled_folder = make_path('01_wrangled_files', args.meteorology_folder)

args.ty_info_folder = make_path('03_ty_info_data', args.working_folder)

args.I_shape = (math.ceil((args.I_x[1]-args.I_x[0])/args.res_degree)+1, math.ceil((args.I_y[1]-args.I_y[0])/args.res_degree)+1)
args.F_shape = (math.ceil((args.F_x[1]-args.F_x[0])/args.res_degree)+1, math.ceil((args.F_y[1]-args.F_y[0])/args.res_degree)+1)
args.O_shape = (math.ceil((args.O_x[1]-args.O_x[0])/args.res_degree)+1, math.ceil((args.O_y[1]-args.O_y[0])/args.res_degree)+1)

args.I_x_iloc = [int((args.I_x[0]-args.O_x[0])/args.res_degree), int((args.I_x[1]-args.O_x[0])/args.res_degree + 1)]
args.I_y_iloc = [int((args.I_y[0]-args.O_y[0])/args.res_degree), int((args.I_y[1]-args.O_y[0])/args.res_degree + 1)]
args.F_x_iloc = [int((args.F_x[0]-args.O_x[0])/args.res_degree), int((args.F_x[1]-args.O_x[0])/args.res_degree + 1)]
args.F_y_iloc = [int((args.F_y[0]-args.O_y[0])/args.res_degree), int((args.F_y[1]-args.O_y[0])/args.res_degree + 1)]

args.xaxis_list = np.around(np.linspace(args.I_x[0], args.I_x[1], args.I_shape[0]), decimals=4)
args.yaxis_list = np.around(np.linspace(args.I_y[1], args.I_y[0], args.I_shape[1]), decimals=4)