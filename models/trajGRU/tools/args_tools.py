# import modules
import os
import math
import argparse
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

parser = argparse.ArgumentParser('')

# arguments for gpu settings

parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA or not. (default=False)')
parser.add_argument('--gpu', default=0, type=int, metavar='',
                    help='Set the gpu device. (default=0)')

# arguments for hyperparams for nn model
parser.add_argument('--max-epochs', default=50, type=int, metavar='',
                    help='Max epochs. (default=50)')
parser.add_argument('--batch-size', default=4, type=int, metavar='',
                    help='Batch size. (default=4)')
parser.add_argument('--lr', default=1e-4, type=float, metavar='',
                    help='Learning rate. (default=1e-4)')
parser.add_argument('--lr-scheduler', action='store_true',
                    help='To set the scheduler for learning rate or not. (default=False)')

parser.add_argument('--weight-decay', default=0.1, type=float, metavar='',
                    help='The factor of weight decay. (default=0.1)')
parser.add_argument('--clip', action='store_true',
                    help='To clip the weightings of the model or not. (default=False)')
parser.add_argument('--clip-max-norm', default=100, type=float, metavar='',
                    help='Max clip norm. (default=100)')

parser.add_argument('--batch-norm', action='store_true',
                    help='To do batch normalization or not. (default=False)')
parser.add_argument('--normalize-target', action='store_true',
                    help='To nomalize target data or not. (default=False)')

working_folder = os.path.expanduser('~/Onedrive/01_IIS/04_TY_research')

parser.add_argument('--working-folder', default=working_folder, type=str,
                    metavar='', help='The path of the mother folder of this code. (the mother folder should have the radar numpy folder, typhoon list excel file, and so on.)')
parser.add_argument('--root-dir', default='01_Radar_data/02_numpy_files', type=str,
                    metavar='', help='The folder path of the Radar numpy data. (default=\'working-folder/01_Radar_data/02_numpy_files\')')
parser.add_argument('--ty-list-file', default='ty_list.xlsx', type=str,
                    metavar='', help='The path of ty_list excel file. (default=\'working-folder/ty_list.xlsx\')')
parser.add_argument('--result-dir', default='04_results', type=str,
                    metavar='', help='The folder path of result files. (default=\'working-folder/04_results\')')
parser.add_argument('--params-dir', default='05_params', type=str,
                    metavar='', help='The folder path of parameter files. (default=\'working-folder/05_params\')')

parser.add_argument('--input-frames', default=6, type=int, metavar='',
                    help='The number of the input frames. (default=6)')
parser.add_argument('--output-frames', default=18, type=int, metavar='',
                    help='The number of the output frames. (default=18)')
parser.add_argument('--input-with-grid', action='store_true',
                    help='To add grid maps into input frames or not. (default=False)')
parser.add_argument('--channel-factor', default=2, type=int, metavar='',
                    help='The channel factor of trajGRU model. (default=2)')

parser.add_argument('--I-lat-l', default=23.9125, type=float, metavar='',
                    help='The lowest latitude of the input frames. (default=23.9125)')
parser.add_argument('--I-lat-h', default=26.15, type=float, metavar='',
                    help='The highest latitude of the input frames. (default=26.15)')
parser.add_argument('--I-lon-l', default=120.4, type=float, metavar='',
                    help='The lowest longitude of the input frames. (default=120.4)')
parser.add_argument('--I-lon-h', default=122.6375, type=float, metavar='',
                    help='The highest longitude of the input frames. (default=122.6375)')

parser.add_argument('--F-lat-l', default=24.6625, type=float, metavar='',
                    help='The lowest latitude of the forecast frames. (default=24.6625)')
parser.add_argument('--F-lat-h', default=25.4, type=float, metavar='',
                    help='The highest latitude of the forecast frames. (default=25.4)')
parser.add_argument('--F-lon-l', default=121.15, type=float, metavar='',
                    help='The lowest longitude of the forecast frames. (default=121.15)')
parser.add_argument('--F-lon-h', default=121.8875, type=float, metavar='',
                    help='The highest longitude of the forecast frames. (default=121.8875)')

parser.add_argument('--res-degree', default=0.0125, type=float, metavar='',
                    help='The res_degree degree of the data. (default=0.0125)')

args = parser.parse_args()

args.working_folder = make_path(args.working_folder)
args.root_dir = make_path(args.root_dir, args.working_folder)
args.ty_list_file = make_path(args.ty_list_file, args.working_folder)
args.result_dir = make_path(args.result_dir, args.working_folder)
args.params_dir = make_path(args.params_dir, args.working_folder)

args.origin_lat_l = 20
args.origin_lat_h = 27
args.origin_lon_l = 118
args.origin_lon_h = 123.5

args.I_x_left = int((args.I_lon_l-args.origin_lon_l)/args.res_degree + 1)
args.I_x_right = int(args.I_x_left + (args.I_lon_h-args.I_lon_l)/args.res_degree + 1)
args.I_y_low = int((args.I_lat_l-args.origin_lat_l)/args.res_degree + 1)
args.I_y_high = int(args.I_y_low + (args.I_lat_h-args.I_lat_l)/args.res_degree + 1)

args.F_x_left = int((args.F_lon_l-args.origin_lon_l)/args.res_degree + 1)
args.F_x_right = int(args.F_x_left + (args.F_lon_h-args.F_lon_l)/args.res_degree + 1)
args.F_y_low = int((args.F_lat_l-args.origin_lat_l)/args.res_degree + 1)
args.F_y_high = int(args.F_y_low + (args.F_lat_h-args.F_lat_l)/args.res_degree + 1)

args.device = None
args.file_shape = None

if not args.disable_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda:{:02d}'.format(args.gpu))
else:
    args.device = torch.device('cpu')

args.F_shape = (math.ceil((args.F_lat_h-args.F_lat_l)/args.res_degree)+1,
                math.ceil((args.F_lon_h-args.F_lon_l)/args.res_degree)+1)
args.I_shape = (math.ceil((args.I_lat_h-args.I_lat_l)/args.res_degree)+1,
                math.ceil((args.I_lon_h-args.I_lon_l)/args.res_degree)+1)

if __name__ == '__main__':
    print(args.input_with_grid)
    print(args.I_shape)
