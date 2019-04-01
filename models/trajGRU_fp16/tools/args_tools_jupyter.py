# import modules
import os
import math
import torch
import pandas as pd
import easydict
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

    if workfolder is not None:
        if not os.path.isabs(new_path):
            return os.path.join(workfolder, new_path)
    return new_path

def remove_file(file):
    if os.path.exists(file):
        os.remove(file)
        
def print_dict(d):
    for key, value in d.items():
        print('{}: {}'.format(key, value))


working_folder = os.path.expanduser('~/ssd/01_ty_research')

radar_folder = make_path('01_radar_data', working_folder)
weather_folder = make_path('02_weather_data', working_folder)
ty_info_folder =  make_path('03_ty_info', working_folder)

args = easydict.EasyDict({
    # define the path of folders
    'working_folder': working_folder,
    'radar_folder': radar_folder,
    'radar_wrangled_data_folder': make_path('02_wrangled_files', radar_folder),
    'weather_folder': weather_folder,
    'weather_wrangled_folder': make_path('01_wrangled_files', weather_folder),
    'ty_info_folder': ty_info_folder,
    'ty_info_wrangled_folder': make_path('01_wrangled_files', ty_info_folder),
    
    'ty_list': make_path('ty_list.csv', working_folder),
    'result_folder': os.path.join(working_folder, '04_results'),
    'params_folder': os.path.join(working_folder, '05_params'),
    
    # control the gpu computation
    'able_cuda': True,
    'gpu': 2,
    'value_dtype': torch.float,
    
    # hyperparameters for training
    'max_epochs':30,
    'batch_size':8,
    'loss_function': BMSE,
    'optimizer': torch.optim.Adam,
    'lr': 1e-4,
    'lr_scheduler': True,
    'weight_decay': 0.1,
    'clip': True,
    'clip_max_norm': 500,
    'batch_norm': True,
    'normalize_target': False,
    'input_frames': 6,
    'target_frames': 18,
    'input_with_grid': True,
    'channel_factor': 2,
    
    # image size
    'res_degree': 0.0125,
    'I_x': [120.9625, 122.075],
    'I_y': [24.4375, 25.55],
    'F_x': [121.3375, 121.7],
    'F_y': [24.8125, 25.175],
    'O_x': [118, 123.5],
    'O_y': [20, 27],
})

if args.able_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda:{:02d}'.format(args.gpu))
else:
    args.device = torch.device('cpu')

args.I_shape = (round((args.I_x[1]-args.I_x[0])/args.res_degree)+1, round((args.I_y[1]-args.I_y[0])/args.res_degree)+1)
args.F_shape = (round((args.F_x[1]-args.F_x[0])/args.res_degree)+1, round((args.F_y[1]-args.F_y[0])/args.res_degree)+1)
args.O_shape = (round((args.O_x[1]-args.O_x[0])/args.res_degree)+1, round((args.O_y[1]-args.O_y[0])/args.res_degree)+1)

# overall info for normalization
rad_overall = pd.read_csv(os.path.join(args.radar_folder, 'overall.csv'), index_col='Measures').loc['max':'min',:]
meteo_overall = pd.read_csv(os.path.join(args.weather_folder, 'overall.csv'), index_col='Measures')
args.max_values = pd.concat([rad_overall, meteo_overall], axis=1, sort=False).T['max']
args.min_values = pd.concat([rad_overall, meteo_overall], axis=1, sort=False).T['min']

# args.I_x_iloc = [int((args.I_x[0]-args.O_x[0])/args.res_degree), int((args.I_x[1]-args.O_x[0])/args.res_degree + 1)]
# args.I_y_iloc = [int((args.I_y[0]-args.O_y[0])/args.res_degree), int((args.I_y[1]-args.O_y[0])/args.res_degree + 1)]
# args.F_x_iloc = [int((args.F_x[0]-args.O_x[0])/args.res_degree), int((args.F_x[1]-args.O_x[0])/args.res_degree + 1)]
# args.F_y_iloc = [int((args.F_y[0]-args.O_y[0])/args.res_degree), int((args.F_y[1]-args.O_y[0])/args.res_degree + 1)]

# args.weather_list = ['PP01', 'PS01', 'RH01', 'TX01','WD01', 'WD02']
args.weather_list = []

args.compression = 'bz2'
args.figure_dpi = 150

args.RAD_level = [-5, 0, 10, 20, 30, 40, 50, 60, 70]
args.QPE_level = [-5, 0, 10, 20, 35, 50, 80, 120, 160, 200]
args.QPF_level = [-5, 0, 10, 20, 35, 50, 80, 120, 160, 200]

args.RAD_cmap = ['#FFFFFF','#FFD8D8','#FFB8B8','#FF9090','#FF6060','#FF2020','#CC0000','#A00000','#600000']
args.QPE_cmap = ['#FFFFFF','#D2D2FF','#AAAAFF','#8282FF','#6A6AFF','#4242FF','#1A1AFF','#000090','#000040','#000030']
args.QPF_cmap = ['#FFFFFF','#D2D2FF','#AAAAFF','#8282FF','#6A6AFF','#4242FF','#1A1AFF','#000090','#000040','#000030']

# args.xaxis_list = np.around(np.linspace(args.I_x[0], args.I_x[1], args.I_shape[0]), decimals=4)
# args.yaxis_list = np.around(np.linspace(args.I_y[1], args.I_y[0], args.I_shape[1]), decimals=4)



if __name__ == '__main__':
    print(args)