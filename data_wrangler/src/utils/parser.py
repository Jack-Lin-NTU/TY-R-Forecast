# import argparse
import os
import numpy as np
import easydict
from .tools import make_path

def get_args():
    working_folder = os.path.expanduser('~/ssd/01_ty_research')

    radar_folder = make_path('01_radar_data', working_folder)
    weather_folder = make_path('02_weather_data', working_folder)
    ty_info_folder =  make_path('03_ty_info', working_folder)

    args = easydict.EasyDict({
        'study_area': 'Taiwan',
        'working_folder': working_folder,
        'ty_list': make_path('ty_list.csv', working_folder),
        'sta_list': make_path('sta_list_all.csv', working_folder),
        'TW_map_file': make_path(os.path.join('07_gis_data','03_TW_shapefile','gadm36_TWN_2'), working_folder),
        'fortran_code_folder': make_path('08_fortran_codes', working_folder),
        # To set the path of radar folders
        'radar_folder': radar_folder,
        'radar_raw_data_folder': make_path(os.path.join('00_raw_data', 'raw_radar_data_2012_2018'), working_folder),
        'radar_compressed_data_folder': make_path('01_compressed_files', radar_folder),
        'radar_wrangled_data_folder': make_path('02_wrangled_files', radar_folder),
        'radar_figures_folder': make_path('03_figures', radar_folder),
        # To set the path of weather data and ty_info folders
        'weather_folder': weather_folder,
        'weather_raw_data_folder': make_path(os.path.join('00_raw_data', 'raw_weather_data_2012_2018'), working_folder),
        'weather_wrangled_data_folder': make_path('01_wrangled_files', weather_folder),
        # To set the path of ty info folders
        'ty_info_folder': ty_info_folder,
        'ty_info_raw_data_folder': make_path(os.path.join('00_raw_data', 'raw_weather_data_2012_2018'), working_folder),
        'ty_info_wrangled_data_folder': make_path('01_wrangled_files', ty_info_folder),
        'res_degree': 0.0125,
        'compression': 'bz2',
        'figure_dpi': 120,
        'RAD_level': [-5, 0, 10, 20, 30, 40, 50, 60, 70, 80],
        'QPE_level': [-5, 0, 10, 20, 35, 50, 80, 120, 160, 200],
        'QPF_level': [-5, 0, 10, 20, 35, 50, 80, 120, 160, 200],
        'PP01_level': [-5, 0, 10, 20, 35, 50, 80, 120, 160],
        'PS01_level': [900, 920, 940, 960, 970, 980, 990, 1000, 1010, 1030],
        'WD01_level': [-5, 0, 5, 10, 15, 20, 30, 40, 50, 60],
        'WD02_level': [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 360],
        'RH01_level': [50, 60, 70, 75, 80, 83, 86, 89, 92, 96, 100],
        'TX01_level': [5, 10, 15, 20, 25, 28, 30, 33, 35],
        'RAD_cmap': ['#FFFFFF','#FFD8D8','#FFB8B8','#FF9090','#FF6060','#FF2020','#CC0000','#A00000','#500000'],
        'QPE_cmap': ['#FFFFFF','#D2D2FF','#AAAAFF','#8282FF','#6A6AFF','#4242FF','#1A1AFF','#000090','#000050','#000030'],
        'QPF_cmap': ['#FFFFFF','#D2D2FF','#AAAAFF','#8282FF','#6A6AFF','#4242FF','#1A1AFF','#000090','#000050','#000030'],
        'PP01_cmap': ['#FFFFFF','#D2D2FF','#AAAAFF','#8282FF','#6A6AFF','#4242FF','#1A1AFF','#000090','#000040'],
        'PS01_cmap': ['#FFFFFF','#FFD8D8','#FFB8B8','#FF9090','#FF6060','#FF2020','#CC0000','#A00000','#600000'],
        'WD01_cmap': ['#FFFFFF','#D8FFD8','#B8FFB8','#90FF90','#60FF60','#20FF20','#00CC00','#00A000','#005000'],
        'WD02_cmap': ['#FFFFFF','#D8FFD8','#B8FFB8','#90FF90','#60FF60','#20FF20','#00CC00','#00A000','#009000','#005000'],
        'RH01_cmap': ['#FFFFFF','#D8D8FF','#B8B8FF','#AAAAFF','#8282FF','#6A6AFF','#4A4AFF','#3A3AFF','#2A2AFF','#000099'],
        'TX01_cmap': ['#FFFFFF','#FFD8D8','#FFB8B8','#FF9090','#FF2020','#CC0000','#A00000','#500000'],
        'weather_names': ['Air pressure(hPa)', r'Temperature($^o$C)', 'Relative huminity(%)', 'Wind speed(m/s)', r'Wind direction($^o$)', 'Precipitation(mm)'],
        'I_x': [120.9625, 122.075],
        'I_y': [24.4375, 25.55],
        'F_x': [121.3375, 121.7],
        'F_y': [24.8125, 25.175],
        'O_y': [20, 27],
        'O_x': [118, 123.5],
    })

    args.I_shape = (round((args.I_x[1]-args.I_x[0])/args.res_degree)+1, round((args.I_y[1]-args.I_y[0])/args.res_degree)+1)
    args.F_shape = (round((args.F_x[1]-args.F_x[0])/args.res_degree)+1, round((args.F_y[1]-args.F_y[0])/args.res_degree)+1)
    args.O_shape = (round((args.O_x[1]-args.O_x[0])/args.res_degree)+1, round((args.O_y[1]-args.O_y[0])/args.res_degree)+1)

    args.I_x_iloc = [int((args.I_x[0]-args.O_x[0])/args.res_degree), int((args.I_x[1]-args.O_x[0])/args.res_degree + 1)]
    args.I_y_iloc = [int((args.I_y[0]-args.O_y[0])/args.res_degree), int((args.I_y[1]-args.O_y[0])/args.res_degree + 1)]
    args.F_x_iloc = [int((args.F_x[0]-args.O_x[0])/args.res_degree), int((args.F_x[1]-args.O_x[0])/args.res_degree + 1)]
    args.F_y_iloc = [int((args.F_y[0]-args.O_y[0])/args.res_degree), int((args.F_y[1]-args.O_y[0])/args.res_degree + 1)]

    args.I_x_list = np.around(np.linspace(args.I_x[0], args.I_x[1], args.I_shape[0]), decimals=4)
    args.I_y_list = np.around(np.linspace(args.I_y[1], args.I_y[0], args.I_shape[1]), decimals=4)
    args.F_x_list = np.around(np.linspace(args.F_x[0], args.F_x[1], args.F_shape[0]), decimals=4)
    args.F_y_list = np.around(np.linspace(args.F_y[1], args.F_y[0], args.F_shape[1]), decimals=4)
    args.O_x_list = np.around(np.linspace(args.O_x[0], args.O_x[1], args.O_shape[0]), decimals=4)
    args.O_y_list = np.around(np.linspace(args.O_y[1], args.O_y[0], args.O_shape[1]), decimals=4)
    
    return args

if __name__ == '__main__':
    args = get_args()
    print(args)