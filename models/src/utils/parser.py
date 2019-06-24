# import modules
import os
import numpy as np
import pandas as pd
import argparse
import torch
import torch.optim as optim

from.utils import make_path, createfolder
from.loss import LOSS

def get_args():
    parser = argparse.ArgumentParser()

    working_folder = os.path.expanduser('~/ssd/01_ty_research')
    radar_folder = make_path('01_radar_data', working_folder)
    weather_folder = make_path('02_weather_data', working_folder)
    ty_info_folder =  make_path('03_ty_info', working_folder)

    parser.add_argument('--working-folder', metavar='', type=str, default=working_folder,
                        help='The path of working folder.(default: ~/ssd/01_ty_research)')
    parser.add_argument('--radar-folder', metavar='', type=str, default=radar_folder,
                        help='The folder path of radar data (relative or absolute).')
    parser.add_argument('--radar-wrangled-data-folder', metavar='', type=str, default=make_path('02_wrangled_files', radar_folder),
                        help='The folder path of  radar data (relative or absolute).')

    parser.add_argument('--weather-folder', metavar='', type=str, default=make_path('02_weather_data', working_folder),
                        help='The folder path of weather data (relative or absolute).')
    parser.add_argument('--weather-wrangled-data-folder', metavar='', type=str, default=make_path('01_wrangled_files', weather_folder),
                        help='The folder path of wrangled weather data (relative or absolute).')

    parser.add_argument('--ty-info-folder', metavar='', type=str, default=make_path('03_ty_info', working_folder),
                    help='The folder path of ty-info data (relative or absolute).')
    parser.add_argument('--ty-info-wrangled-data-folder', metavar='', type=str, default=make_path('01_wrangled_files', ty_info_folder),
                    help='The folder path of wrangled ty-info data (relative or absolute).')

    parser.add_argument('--result-folder', metavar='', type=str, default=make_path('04_results', working_folder),
                    help='The path of result folder.')
    parser.add_argument('--params-folder', metavar='', type=str, default=make_path('05_params', working_folder),
                    help='The path of params folder.')
    parser.add_argument('--infers-folder', metavar='', type=str, default=make_path('06_inferences', working_folder),
                    help='The path of params folder.')

    parser.add_argument('--ty-list', metavar='', type=str, default=make_path('ty_list.csv', working_folder),
                    help='The file path of ty-list.csv.')

    parser.add_argument('--model', metavar='', type=str, default='trajGRU', help='The GRU model applied here. (default: TRAJGRU)')
    parser.add_argument('--parallel-compute', action='store_true', help='Parallel computing.')
    parser.add_argument('--able-cuda', action='store_true', help='Able cuda. (default: disable cuda)')
    parser.add_argument('--gpu', metavar='', type=int, default=0, help='GPU device. (default: 0)')
    parser.add_argument('--value-dtype', metavar='', type=str, default='float32', help='The data type of computation. (default: float32)')
    parser.add_argument('--change-value-dtype', action='store_true', help='Change the data type of computation.')

    # hyperparameters for training
    parser.add_argument('--seed', metavar='', type=int, default=1, help='The setting of random seed. (default: 1)')
    parser.add_argument('--train-num', metavar='', type=int, default=10, help='The number of training events. (default: 10)')
    parser.add_argument('--max-epochs', metavar='', type=int, default=30, help='Max epochs. (default: 30)')
    parser.add_argument('--batch-size', metavar='', type=int, default=4, help='Batch size. (default: 8)')
    parser.add_argument('--lr', metavar='', type=float, default=1e-3, help='Learning rate. (default: 1e-3)')
    parser.add_argument('--lr-scheduler', action='store_true', help='Set a scheduler for controlling learning rate.')
    parser.add_argument('--weight-decay', metavar='', type=float, default=0, help='The value setting of wegiht decay. (default: 0)')
    parser.add_argument('--clip', action='store_true', help='Clip the weightings in the model.')
    parser.add_argument('--clip-max-norm', metavar='', type=float, default=10, help='Max norm value for clipping weightings. (default: 1)')
    parser.add_argument('--batch-norm', action='store_true', help='Do batch normalization.')
    parser.add_argument('--normalize-input', action='store_true', help='Normalize inputs.')
    parser.add_argument('--normalize-target', action='store_true', help='Normalize targets.')

    parser.add_argument('--optimizer', metavar='', type=str, default='Adam', help='The optimizer. (default: Adam)')
    parser.add_argument('--loss-function', metavar='', type=str, default='BMSE', help='The loss function. (default: BMSE)')
    parser.add_argument('--input-frames', metavar='', type=int, default=6, help='The size of input frames. (default: 6)')
    parser.add_argument('--target-frames', metavar='', type=int, default=18, help='The size of target frames. (default: 18)')
    parser.add_argument('--input-with-grid', action='store_true', help='Input with grid data.')
    parser.add_argument('--input-with-QPE', action='store_true', help='Input with QPE data.')
    parser.add_argument('--target-RAD', action='store_true', help='Use RAD-transformed data as targets.')
    parser.add_argument('--denoise-RAD', action='store_true', help='Use denoised RAD data as inputs.')
    parser.add_argument('--channel-factor', metavar='', type=int, default=2, help='Channel factor. (default: 2)')
    parser.add_argument('--multi-unit', action='store_true', help='Use multi-unit.')

    # hyperparameters for STN-CONVGRU
    parser.add_argument('--catcher-location', action='store_true', help='Input only location info of typhoon.')

    # tw forecast size (400x400)
    parser.add_argument('--I-size', metavar='', type=int, default=420, help='The height of inputs (default: 420)')
    parser.add_argument('--F-size', metavar='', type=int, default=420, help='The height of targets (default: 420)')

    parser.add_argument('--weather-list', metavar='', action='append', default=[],
                        help='Weather list. (default: [])')

    # Announce the args
    args = parser.parse_args()
    # Adjust some settings in the args.
    if args.able_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda:{:02d}'.format(args.gpu))
    else:
        args.device = torch.device('cpu')
    args.value_dtype = getattr(torch, args.value_dtype)


    # variables for locating maps.
    args.map_center = [120.75, 23.5]
    args.res_degree = 0.0125
    args.O_x = [118, 123.5]
    args.O_y = [20, 27]
    args.I_x = [args.map_center[0]-(args.res_degree*args.I_size/2), args.map_center[0]+(args.res_degree*(args.I_size/2-1))]
    args.I_y = [args.map_center[1]-(args.res_degree*args.I_size/2), args.map_center[1]+(args.res_degree*(args.I_size/2-1))]
    args.F_x = [args.map_center[0]-(args.res_degree*args.F_size/2), args.map_center[0]+(args.res_degree*(args.F_size/2-1))]
    args.F_y = [args.map_center[1]-(args.res_degree*args.F_size/2), args.map_center[1]+(args.res_degree*(args.F_size/2-1))]

    args.I_shape = (args.I_size, args.I_size)
    args.F_shape = (args.F_size, args.F_size)
    args.O_shape = (441, 561)

    args.I_x_iloc = [int((args.I_x[0]-args.O_x[0])/args.res_degree), 441-int((args.O_x[1]-args.I_x[1])/args.res_degree)]
    args.F_x_iloc = [int((args.F_x[0]-args.O_x[0])/args.res_degree), 441-int((args.O_x[1]-args.F_x[1])/args.res_degree)]

    args.I_y_iloc = [int((args.I_y[0]-args.O_y[0])/args.res_degree), int(561-(args.O_y[1]-args.I_y[1])/args.res_degree)]
    args.F_y_iloc = [int((args.F_y[0]-args.O_y[0])/args.res_degree), int(561-(args.O_y[1]-args.F_y[1])/args.res_degree)]

    args.I_x_list = np.around(np.linspace(args.I_x[0], args.I_x[1], args.I_shape[0]), decimals=4)
    args.I_y_list = np.around(np.linspace(args.I_y[1], args.I_y[0], args.I_shape[1]), decimals=4)

    # statistics of each data
    rad_overall = pd.read_csv(os.path.join(args.radar_folder, 'overall.csv'), index_col='Measures').loc['max_value':'min_value',:]
    if len(args.weather_list) == 0:
        meteo_overall = None
    else:
        meteo_overall = pd.read_csv(os.path.join(args.weather_folder, 'overall.csv'), index_col='Measures')
    ty_overall = pd.read_csv(os.path.join(args.ty_info_folder, 'overall.csv'), index_col=0).T
    if args.catcher_location:
        ty_overall = ty_overall[['Lat','Lon','distance']]
    else:
        ty_overall = ty_overall.iloc[:,0:-1]

    args.max_values = pd.concat([rad_overall, meteo_overall, ty_overall], axis=1, sort=False).T['max_value']
    args.min_values = pd.concat([rad_overall, meteo_overall, ty_overall], axis=1, sort=False).T['min_value']

    # define loss function
    args.loss_function = LOSS(args=args)

    args.compression = 'bz2'
    args.figure_dpi = 120

    args.RAD_level = [-5, 0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    args.QPE_level = [-5, 0, 5, 10, 20, 35, 50, 80, 100, 200]
    args.QPF_level = [-5, 0, 5, 10, 20, 35, 50, 80, 100, 200]
    args.RAD_cmap = ['#FFFFFF','#FFD8D8','#FFB8B8','#FF9090','#FF6060','#FF2020','#CC0000','#A00000','#600000','#450000','#300000']
    args.QPE_cmap = ['#FFFFFF','#D2D2FF','#AAAAFF','#8282FF','#6A6AFF','#4242FF','#1A1AFF','#000090','#000050','#000030']
    args.QPF_cmap = ['#FFFFFF','#D2D2FF','#AAAAFF','#8282FF','#6A6AFF','#4242FF','#1A1AFF','#000090','#000050','#000030']

    args.input_channels = 1 + args.input_with_QPE*1 + len(args.weather_list) + args.input_with_grid*2

    args.TW_map_file = make_path(os.path.join('07_gis_data','03_TW_shapefile','gadm36_TWN_2'), working_folder)

    # make folders' path
    args.result_folder = os.path.join(args.result_folder, args.model.upper())
    args.params_folder = os.path.join(args.params_folder, args.model.upper())

    size = '{}X{}'.format(args.F_shape[0], args.F_shape[1])

    if args.weather_list == []:
        args.result_folder = os.path.join(args.result_folder, 'RAD')
        args.params_folder = os.path.join(args.params_folder, 'RAD')
    else:
        args.result_folder = os.path.join(args.result_folder, 'RAD_weather')
        args.params_folder = os.path.join(args.params_folder, 'RAD_weather')
    
    if args.input_with_grid:
        args.result_folder += '_grid'
        args.params_folder += '_grid'

    if args.input_with_QPE:
        args.result_folder += '_QPE'
        args.params_folder += '_QPE'
    
    if args.lr_scheduler:
        args.result_folder += '_scheduler'
        args.params_folder += '_scheduler'

    args.result_folder += '_'+args.optimizer
    args.params_folder += '_'+args.optimizer

    if args.weight_decay==0:
        args.result_folder += 'lr{:.5f}'.format(args.weight_decay, args.lr)
        args.params_folder += 'lr{:.5f}'.format(args.weight_decay, args.lr)
    else:
        args.result_folder += 'wd{:.5f}_lr{:.5f}'.format(args.weight_decay, args.lr)
        args.params_folder += 'wd{:.5f}_lr{:.5f}'.format(args.weight_decay, args.lr)

    if args.clip:
        args.result_folder += '_clip'+str(args.clip_max_norm)
        args.params_folder += '_clip'+str(args.clip_max_norm)

    if args.catcher_location:
        args.result_folder += '_chatchloc'
        args.params_folder += '_chatchloc'

    # denoised data as inputs
    if args.denoise_RAD:
        args.radar_wrangled_data_folder += '_denoise20'

    createfolder(args.result_folder)
    createfolder(args.params_folder)
    return args

def print_args(args):
    with open(os.path.join(args.result_folder, 'hyperparams.txt'), 'w') as f:
        f.write('Device: {}\n'.format(args.device))
        f.write('Model: {}\n'.format(args.model))
        f.write('Optimizer: {}\n'.format(args.optimizer))
        f.write('Lr: {}\n'.format(args.lr))
        f.write('Lr schedule: {}\n'.format(args.lr_scheduler))
        f.write('Max epochs: {}\n'.format(args.max_epochs))
        f.write('Batch size: {}\n'.format(args.batch_size))
        f.write('Clip: {}\n'.format(args.clip))
        f.write('Clipping norm: {}\n'.format(args.clip_max_norm))
        f.write('Batch normalize: {}\n'.format(args.batch_norm))
        f.write('Normalize Target: {}\n'.format(args.normalize_target))
        f.write('Normalize Input: {}\n'.format(args.normalize_input))
        f.write('Denoised RAD: {}\n'.format(args.denoise_RAD))
        f.write('Input channels: {}\n'.format(args.input_channels))
        f.write('Input frames: {}\n'.format(args.input_frames))
        f.write('Input with QPE: {}\n'.format(args.input_with_QPE))
        f.write('Input with grid: {}\n'.format(args.input_with_grid))
        f.write('Multi-units: {}\n'.format(args.multi_unit))