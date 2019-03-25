# import modules
import os
import math
import easydict
import torch

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

working_folder = os.path.expanduser('~/Onedrive/01_IIS/04_TY_research')

args = easydict.EasyDict({
    # control the gpu computation
    'disable_cuda': False,
    'gpu': 0,
    # hyperparameters for training
    'max_epochs':50,
    'lr': 1e-4,
    'lr_scheduler': False,
    'weight_decay': 0.1,
    'clip': False,
    'clip_max_norm': 100,
    'batch_norm': False,
    'normalize_target': False,
    'input_frames': 6,
    'output_frames': 18,
    'input_with_grid': False,
    'channel_factor': 2,
    'I_y': [23.9125, 26.15],
    'I_x': [120.4, 122.6375],
    'F_y': [24.6625, 25.4],
    'F_x': [121.15, 121.8875],
    'res_degree': 0.0125,
    'O_y': [20, 27],
    'O_x': [118,123.5],
    # define the path of folders
    'working_folder': working_folder,
    'root_dir': os.path.join(working_folder, '01_Radar_data/02_numpy_files'),
    'ty_list_file': os.path.join(working_folder, 'ty_list.xlsx'),
    'result_dir': os.path.join(working_folder, '04_results'),
    'params_dir': os.path.join(working_folder, '05_params'),
})

args.I_x_left = int((args.I_lon_l-args.origin_lon_l)/args.res_degree + 1)
args.I_x_right = int(args.I_x_left + (args.I_lon_h-args.I_lon_l)/args.res_degree + 1)
args.I_y_low = int((args.I_lat_l-args.origin_lat_l)/args.res_degree + 1)
args.I_y_high = int(args.I_y_low + (args.I_lat_h-args.I_lat_l)/args.res_degree + 1)

args.F_x_left = int((args.F_lon_l-args.origin_lon_l)/args.res_degree + 1)
args.F_x_right = int(args.F_x_left + (args.F_lon_h-args.F_lon_l)/args.res_degree + 1)
args.F_y_low = int((args.F_lat_l-args.origin_lat_l)/args.res_degree + 1)
args.F_y_high = int(args.F_y_low + (args.F_lat_h-args.F_lat_l)/args.res_degree + 1)


if not args.disable_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda:{:02d}'.format(args.gpu))
else:
    args.device = torch.device('cpu')

args.F_shape = (math.ceil((args.F_lat_h-args.F_lat_l)/args.res_degree)+1,
                math.ceil((args.F_lon_h-args.F_lon_l)/args.res_degree)+1)
args.I_shape = (math.ceil((args.I_lat_h-args.I_lat_l)/args.res_degree)+1,
                math.ceil((args.I_lon_h-args.I_lon_l)/args.res_degree)+1)
