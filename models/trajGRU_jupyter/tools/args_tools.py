# import modules
import os
import math
import torch
import pandas as pd
import argparse
from .loss_function import BMSE, BMAE

def createfolder(directory):
    '''
    This function is used to create new folder with given directory.
    '''
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' +  directory)

def make_path(path, workfolder=None):
    '''
    This function is to make absolute path.
    '''
    if path[0] == '~':
        new_path = os.path.expanduser(path)
    else:
        new_path = path

    if workfolder is not None and not os.path.isabs(new_path):
        return os.path.join(workfolder, new_path)
        
    return new_path

def remove_file(file):
    if os.path.exists(file):
        os.remove(file)
        
def print_dict(d):
    for key, value in d.items():
        print('{}: {}'.format(key, value))



parser = argparse.ArgumentParser()

working_folder = os.path.expanduser('~/ssd/01_ty_research')
radar_folder = make_path('01_radar_data', working_folder)
weather_folder = make_path('02_weather_data', working_folder)
ty_info_folder =  make_path('03_ty_info', working_folder)

parser.add_argument('--working-folder', metavar='', type=str, default=working_folder,
                   help='The path of working folder.')
parser.add_argument('--radar-folder', metavar='', type=str, default=radar_folder,
                   help='The path of radar folder.')
parser.add_argument('--radar-wrangled-data-folder', metavar='', type=str, default=make_path('02_wrangled_files', radar_folder),
                   help='The path of radar wrangled-data folder.')

parser.add_argument('--weather-folder', metavar='', type=str, default=make_path('02_weather_data', working_folder),
                   help='The path of weather folder.')
parser.add_argument('--weather-wrangled-data-folder', metavar='', type=str, default=make_path('01_wrangled_files', weather_folder),
                   help='The path of weather wrangled-data folder.')

parser.add_argument('--ty-info-folder', metavar='', type=str, default=make_path('03_ty_info', working_folder),
                   help='The path of ty-info folder.')
parser.add_argument('--ty-info-wrangled-data-folder', metavar='', type=str, default=make_path('01_wrangled_files', ty_info_folder),
                   help='The path of ty-info wrangled-data folder.')


parser.add_argument('--result-folder', metavar='', type=str, default=make_path('04_results', working_folder),
                   help='The path of result folder.')
parser.add_argument('--params-folder', metavar='', type=str, default=make_path('05_params', working_folder),
                   help='The path of params folder.')

parser.add_argument('--ty-list', metavar='', type=str, default=make_path('ty_list.csv', working_folder),
                   help='The path of ty list file.')

parser.add_argument('--able-cuda', action='store_true', help='Able cuda.')
parser.add_argument('--gpu', metavar='', type=int, default=2, help='GPU device.(default: 2)')

# hyperparameters for training
parser.add_argument('--max-epochs', metavar='', type=int, default=30, help='Max epochs.(default: 30)')
parser.add_argument('--batch-size', metavar='', type=int, default=8, help='Batch size.(default: 8)')
parser.add_argument('--lr', metavar='', type=float, default=1e-4, help='Max epochs.(default: 1e-4)')
parser.add_argument('--lr-scheduler', action='store_true', help='Set lr-scheduler.')
parser.add_argument('--weight-decay', metavar='', type=float, default=0.1, help='Wegiht decay.(default: 0.1)')
parser.add_argument('--clip', action='store_true', help='Clip model weightings.')
parser.add_argument('--clip-max-norm', metavar='', type=int, default=500, help='Max norm value for clip model weightings. (default: 500)')
parser.add_argument('--batch-norm', action='store_true', help='Do batch normalization.')

parser.add_argument('--normalize-target', action='store_true', help='Normalize target maps.')

parser.add_argument('--input-frames', metavar='', type=int, default=6, help='The size of input frames. (default: 6)')
parser.add_argument('--input-with-grid', action='store_true', help='Input with grid data.')
parser.add_argument('--input-with-QPE', action='store_true', help='Input with QPE data.')
parser.add_argument('--target-frames', metavar='', type=int, default=18, help='The size of target frames. (default: 18)')
parser.add_argument('--channel-factor', metavar='', type=int, default=3, help='Channel factor. (default: 3)')

parser.add_argument('--I-x-l', metavar='', type=float, default=120.9625, help='The lowest longitude of input map. (default: 120.9625)')
parser.add_argument('--I-x-h', metavar='', type=float, default=122.075, help='The highest longitude of input map. (default: 122.075)')
parser.add_argument('--I-y-l', metavar='', type=float, default=24.4375, help='The lowest latitude of input map. (default: 24.4375)')
parser.add_argument('--I-y-h', metavar='', type=float, default=25.55, help='The highest latitude of input map. (default: 25.55)')

parser.add_argument('--F-x-l', metavar='', type=float, default=121.3375, help='The lowest longitude of target map. (default: 121.3375)')
parser.add_argument('--F-x-h', metavar='', type=float, default=121.7, help='The highest longitude of target map. (default: 121.7)')
parser.add_argument('--F-y-l', metavar='', type=float, default=24.8125, help='The lowest latitude of target map. (default: 24.8125)')
parser.add_argument('--F-y-h', metavar='', type=float, default=25.175, help='The highest latitude of target map. (default: 25.175)')

parser.add_argument('--O-x-l', metavar='', type=float, default=118, help='The lowest longitude of original map. (default: 118)')
parser.add_argument('--O-x-h', metavar='', type=float, default=123.5, help='The highest longitude of original map. (default: 123.5)')
parser.add_argument('--O-y-l', metavar='', type=float, default=20, help='The lowest latitude of original map. (default: 20)')
parser.add_argument('--O-y-h', metavar='', type=float, default=27, help='The highest latitude of original map. (default: 27)')

parser.add_argument('--weather-list', metavar='', action='append', default=[],
                    help='Weather list. (default: [])')


args = parser.parse_args()

if args.able_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda:{:02d}'.format(args.gpu))
else:
    args.device = torch.device('cpu')

args.value_dtype = torch.float
args.loss_function = BMSE
args.optimizer = torch.optim.Adam

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
meteo_overall = pd.read_csv(os.path.join(args.weather_folder, 'overall.csv'), index_col='Measures')
args.max_values = pd.concat([rad_overall, meteo_overall], axis=1, sort=False).T['max']
args.min_values = pd.concat([rad_overall, meteo_overall], axis=1, sort=False).T['min']

# # args.I_x_iloc = [int((args.I_x[0]-args.O_x[0])/args.res_degree), int((args.I_x[1]-args.O_x[0])/args.res_degree + 1)]
# # args.I_y_iloc = [int((args.I_y[0]-args.O_y[0])/args.res_degree), int((args.I_y[1]-args.O_y[0])/args.res_degree + 1)]
# # args.F_x_iloc = [int((args.F_x[0]-args.O_x[0])/args.res_degree), int((args.F_x[1]-args.O_x[0])/args.res_degree + 1)]
# # args.F_y_iloc = [int((args.F_y[0]-args.O_y[0])/args.res_degree), int((args.F_y[1]-args.O_y[0])/args.res_degree + 1)]


args.compression = 'bz2'
args.figure_dpi = 150

args.RAD_level = [-5, 0, 10, 20, 30, 40, 50, 60, 70]
args.QPE_level = [-5, 0, 10, 20, 35, 50, 80, 120, 160, 200]
args.QPF_level = [-5, 0, 10, 20, 35, 50, 80, 120, 160, 200]

args.RAD_cmap = ['#FFFFFF','#FFD8D8','#FFB8B8','#FF9090','#FF6060','#FF2020','#CC0000','#A00000','#600000']
args.QPE_cmap = ['#FFFFFF','#D2D2FF','#AAAAFF','#8282FF','#6A6AFF','#4242FF','#1A1AFF','#000090','#000040','#000030']
args.QPF_cmap = ['#FFFFFF','#D2D2FF','#AAAAFF','#8282FF','#6A6AFF','#4242FF','#1A1AFF','#000090','#000040','#000030']

# # args.xaxis_list = np.around(np.linspace(args.I_x[0], args.I_x[1], args.I_shape[0]), decimals=4)
# # args.yaxis_list = np.around(np.linspace(args.I_y[1], args.I_y[0], args.I_shape[1]), decimals=4)



if __name__ == '__main__':
    print(args.lr)
