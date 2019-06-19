# import modules
import os
import numpy as np
import pandas as pd
import argparse
import torch
import torch.optim as optim

from.utils import make_path
from.loss import MAE, MSE

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
    parser.add_argument('--able-cuda', action='store_true', help='Able cuda. (default: disable cuda)')
    parser.add_argument('--gpu', metavar='', type=int, default=0, help='GPU device. (default: 0)')
    parser.add_argument('--value-dtype', metavar='', type=str, default='float32', help='The data type of computation. (default: float32)')
    parser.add_argument('--change-value-dtype', action='store_true', help='Change the data type of computation.')

    # hyperparameters for training
    parser.add_argument('--parallel-compute', action='store_true', help='Parallel computing.')
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
    parser.add_argument('--normalize-target', action='store_true', help='Normalize targets.')

    parser.add_argument('--optimizer', metavar='', type=str, default='Adam', help='The optimizer. (default: Adam)')
    parser.add_argument('--loss-function', metavar='', type=str, default='BMSE', help='The loss function. (default: BMSE)')
    parser.add_argument('--input-frames', metavar='', type=int, default=6, help='The size of input frames. (default: 6)')
    parser.add_argument('--target-frames', metavar='', type=int, default=18, help='The size of target frames. (default: 18)')
    parser.add_argument('--input-with-grid', action='store_true', help='Input with grid data.')
    parser.add_argument('--input-with-QPE', action='store_true', help='Input with QPE data.')
    parser.add_argument('--target-RAD', action='store_true', help='Use RAD-transformed data as targets.')
    parser.add_argument('--channel-factor', metavar='', type=int, default=2, help='Channel factor. (default: 2)')

    # parser.add_argument('--I-x-l', metavar='', type=float, default=120.9625, help='The lowest longitude of input map. (default: 120.9625)')
    # parser.add_argument('--I-x-h', metavar='', type=float, default=122.075, help='The highest longitude of input map. (default: 122.075)')
    # parser.add_argument('--I-y-l', metavar='', type=float, default=24.4375, help='The lowest latitude of input map. (default: 24.4375)')
    # parser.add_argument('--I-y-h', metavar='', type=float, default=25.55, help='The highest latitude of input map. (default: 25.55)')

    # parser.add_argument('--F-x-l', metavar='', type=float, default=121.3375, help='The lowest longitude of target map. (default: 121.3375)')
    # parser.add_argument('--F-x-h', metavar='', type=float, default=121.7, help='The highest longitude of target map. (default: 121.7)')
    # parser.add_argument('--F-y-l', metavar='', type=float, default=24.8125, help='The lowest latitude of target map. (default: 24.8125)')
    # parser.add_argument('--F-y-h', metavar='', type=float, default=25.175, help='The highest latitude of target map. (default: 25.175)')

    # tw forecast size (400x400)
    parser.add_argument('--I-x-l', metavar='', type=float, default=118.3, help='The lowest longitude of input map. (default: 118.3)')
    parser.add_argument('--I-x-h', metavar='', type=float, default=123.2875, help='The highest longitude of input map. (default: 123.2875)')
    parser.add_argument('--I-y-l', metavar='', type=float, default=21, help='The lowest latitude of input map. (default: 21)')
    parser.add_argument('--I-y-h', metavar='', type=float, default=25.9875, help='The highest latitude of input map. (default: 25.9875)')

    parser.add_argument('--F-x-l', metavar='', type=float, default=118.3, help='The lowest longitude of target map. (default: 118.3)')
    parser.add_argument('--F-x-h', metavar='', type=float, default=123.2875, help='The highest longitude of target map. (default: 123.2875)')
    parser.add_argument('--F-y-l', metavar='', type=float, default=21, help='The lowest latitude of target map. (default: 21)')
    parser.add_argument('--F-y-h', metavar='', type=float, default=25.9875, help='The highest latitude of target map. (default: 25.9875)')

    parser.add_argument('--O-x-l', metavar='', type=float, default=118, help='The lowest longitude of original map. (default: 118)')
    parser.add_argument('--O-x-h', metavar='', type=float, default=123.5, help='The highest longitude of original map. (default: 123.5)')
    parser.add_argument('--O-y-l', metavar='', type=float, default=20, help='The lowest latitude of original map. (default: 20)')
    parser.add_argument('--O-y-h', metavar='', type=float, default=27, help='The highest latitude of original map. (default: 27)')

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

    args.res_degree = 0.0125
    args.I_x = [args.I_x_l, args.I_x_h]
    args.I_y = [args.I_y_l, args.I_y_h]
    args.F_x = [args.F_x_l, args.F_x_h]
    args.F_y = [args.F_y_l, args.F_y_h]
    args.O_x = [args.O_x_l, args.O_x_h]
    args.O_y = [args.O_y_l, args.O_y_h]

    args.I_shape = (round((args.I_x_h-args.I_x_l)/args.res_degree)+1, round((args.I_y_h-args.I_y_l)/args.res_degree)+1)
    args.F_shape = (round((args.F_x_h-args.F_x_l)/args.res_degree)+1, round((args.F_y_h-args.F_y_l)/args.res_degree)+1)
    args.O_shape = (round((args.O_x_h-args.O_x_l)/args.res_degree)+1, round((args.O_y_h-args.O_y_l)/args.res_degree)+1)

    # overall info for normalization
    rad_overall = pd.read_csv(os.path.join(args.radar_folder, 'overall.csv'), index_col='Measures').loc['max':'min',:]

    if len(args.weather_list) == 0:
        meteo_overall = None
    else:
        meteo_overall = pd.read_csv(os.path.join(args.weather_folder, 'overall.csv'), index_col='Measures')
    
    ty_overall = pd.read_csv(os.path.join(args.ty_info_folder, 'overall.csv'), index_col=0).T

    args.max_values = pd.concat([rad_overall, meteo_overall, ty_overall], axis=1, sort=False).T['max']
    args.min_values = pd.concat([rad_overall, meteo_overall, ty_overall], axis=1, sort=False).T['min']

    if args.loss_function.upper() == 'BMSE':
        args.loss_function = MSE(max_values=args.max_values['QPE'], min_values=args.min_values['QPE'], balance=True, normalize_target=args.normalize_target)
    elif args.loss_function.upper() == 'BMAE':
        args.loss_function = MAE(max_values=args.max_values['QPE'], min_values=args.min_values['QPE'], balance=True, normalize_target=args.normalize_target)
    elif args.loss_function.upper() == 'MSE':
        args.loss_function = MSE(max_values=args.max_values['QPE'], min_values=args.min_values['QPE'], balance=False, normalize_target=args.normalize_target)
    elif args.loss_function.upper() == 'MAE':
        args.loss_function = MAE(max_values=args.max_values['QPE'], min_values=args.min_values['QPE'], balance=True, normalize_target=args.normalize_target)

    args.I_x_iloc = [int((args.I_x[0]-args.O_x[0])/args.res_degree), int((args.I_x[1]-args.O_x[0])/args.res_degree + 1)]
    args.I_y_iloc = [int((args.I_y[0]-args.O_y[0])/args.res_degree), int((args.I_y[1]-args.O_y[0])/args.res_degree + 1)]
    args.F_x_iloc = [int((args.F_x[0]-args.O_x[0])/args.res_degree), int((args.F_x[1]-args.O_x[0])/args.res_degree + 1)]
    args.F_y_iloc = [int((args.F_y[0]-args.O_y[0])/args.res_degree), int((args.F_y[1]-args.O_y[0])/args.res_degree + 1)]

    args.I_x_list = np.around(np.linspace(args.I_x[0], args.I_x[1], args.I_shape[0]), decimals=4)
    args.I_y_list = np.around(np.linspace(args.I_y[1], args.I_y[0], args.I_shape[1]), decimals=4)

    args.compression = 'bz2'
    args.figure_dpi = 120

    args.RAD_level = [-5, 0, 10, 20, 30, 40, 50, 60, 70]
    args.QPE_level = [-5, 0, 5, 10, 20, 35, 50, 80, 100, 200]
    args.QPF_level = [-5, 0, 10, 20, 35, 50, 80, 120, 160, 200]

    args.RAD_cmap = ['#FFFFFF','#FFD8D8','#FFB8B8','#FF9090','#FF6060','#FF2020','#CC0000','#A00000','#600000']
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
    

    return args

