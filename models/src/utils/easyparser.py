# This parser file is for juypter notebook.
import os
import numpy as np
import pandas as pd
from easydict import EasyDict as edict
import torch
import torch.optim as optim

from.utils import make_path, createfolder
from.loss import Loss


class parser():
	def __init__(self):
		working_folder = os.path.expanduser('~/ssd/01_ty_research')
		
		self.initial_args = edict({
			# Path setting
			'working_folder': working_folder,
			'ty_list': os.path.join(working_folder, 'ty_list.csv'),
			'radar_folder': os.path.join(working_folder, '01_radar_data'),
			'weather_folder': os.path.join(working_folder, '02_weather_data'),
			'ty_info_folder': os.path.join(working_folder, '03_ty_info'),
			'radar_wrangled_data_folder': os.path.join(working_folder, '01_radar_data', '02_wrangled_files_smooth'),
			'weather_wrangled_data_folder': os.path.join(working_folder, '02_weather_data', '01_wrangled_files'),
			'ty_info_wrangled_data_folder': os.path.join(working_folder, '03_ty_info', '01_wrangled_files'),
			'result_folder': os.path.join(working_folder, '04_results'),
			'params_folder': os.path.join(working_folder, '05_params'),
			'infers_folder': os.path.join(working_folder, '06_inferences'),
			# Device setting
			'able_cuda': True,
			'gpu': 0,
			'value_dtype': 'float32',
			# Hyperparameter setting
			'seed':1000 ,
			'channel_factor': 2,	# for RNN-based model
			'random_seed': 1000,
			'train_num': 10,
			'max_epochs': 30,
			'batch_size': 8,
			'lr': 0.0001,
			'lr_scheduler': True,
			'weight_decay': 0.01,
			'clip': True,
			'clip_max_norm': 1,
			'batch_norm': True,
			'normalize_input': False,
			'normalize_target': False,
			'optimizer': 'Adam',
			'loss_function': 'BMSE',
			'model': 'Transformer',
			# Data setting
			'weather_list': [],
			'I_nframes': 6,
			'F_nframes': 18,
			'I_size': 400,
			'F_size': 400,
			'input_with_grid': False,
			'input_with_height': False,
			'flow_enable': False,
			'flow_nframes': 2,
			'denoise_RAD': False,
			'loc_catcher': 8, # for STN-CONVGRU model
			'map_center': [120.75, 23.5],
			'res_degree': 0.0125,
			'O_x': [118, 123.5],
			'O_y': [20, 27],
			'O_shape': (441, 561)
			})
	
	def get_args(self):
		args = self.initial_args

		if args.able_cuda and torch.cuda.is_available():
			args.device = torch.device('cuda:{:02d}'.format(args.gpu))
		else:
			args.device = torch.device('cpu')
		args.value_dtype = getattr(torch, args.value_dtype)

		args.I_shape = (args.I_size, args.I_size)
		args.F_shape = (args.F_size, args.F_size)

		args.I_x = np.array([args.map_center[0]-(args.res_degree*args.I_size/2), args.map_center[0]+(args.res_degree*(args.I_size/2-1))])
		args.I_y = np.array([args.map_center[1]-(args.res_degree*args.I_size/2), args.map_center[1]+(args.res_degree*(args.I_size/2-1))])
		args.F_x = np.array([args.map_center[0]-(args.res_degree*args.F_size/2), args.map_center[0]+(args.res_degree*(args.F_size/2-1))])
		args.F_y = np.array([args.map_center[1]-(args.res_degree*args.F_size/2), args.map_center[1]+(args.res_degree*(args.F_size/2-1))])

		args.I_x_iloc = np.array([int((args.I_x[0]-args.O_x[0])/args.res_degree), 441-int((args.O_x[1]-args.I_x[1])/args.res_degree)])
		args.F_x_iloc = np.array([int((args.F_x[0]-args.O_x[0])/args.res_degree), 441-int((args.O_x[1]-args.F_x[1])/args.res_degree)])
		args.I_y_iloc = np.array([int((args.I_y[0]-args.O_y[0])/args.res_degree), int(561-(args.O_y[1]-args.I_y[1])/args.res_degree)])
		args.F_y_iloc = np.array([int((args.F_y[0]-args.O_y[0])/args.res_degree), int(561-(args.O_y[1]-args.F_y[1])/args.res_degree)])

		args.I_x_list = np.around(np.linspace(args.I_x[0], args.I_x[1], args.I_shape[0]), decimals=4)
		args.I_y_list = np.around(np.linspace(args.I_y[1], args.I_y[0], args.I_shape[1]), decimals=4)

		# statistics of each data
		rad_overall = pd.read_csv(os.path.join(args.radar_folder, 'overall.csv'), index_col='Measures').loc['max_value':'min_value',:]

		if len(args.weather_list) == 0:
			meteo_overall = None
		else:
			meteo_overall = pd.read_csv(os.path.join(args.weather_folder, 'overall.csv'), index_col='Measures')

		ty_overall = pd.read_csv(os.path.join(args.ty_info_folder, 'overall.csv'), index_col=0).T
		if args.loc_catcher:
			ty_overall = ty_overall[['Lat','Lon','distance']]
		else:
			ty_overall = ty_overall.iloc[:,0:-1]

		args.max_values = pd.concat([rad_overall, meteo_overall, ty_overall], axis=1, sort=False).T['max_value']
		args.min_values = pd.concat([rad_overall, meteo_overall, ty_overall], axis=1, sort=False).T['min_value']

		# define loss function
		args.loss_function = Loss(args=args)

		args.compression = 'bz2'
		args.figure_dpi = 120

		args.RAD_level = [-5, 0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
		args.QPE_level = [-5, 0, 5, 10, 20, 35, 50, 80, 100, 200]
		args.QPF_level = [-5, 0, 5, 10, 20, 35, 50, 80, 100, 200]
		args.RAD_cmap = ['#FFFFFF','#FFD8D8','#FFB8B8','#FF9090','#FF6060','#FF2020','#CC0000','#A00000','#600000','#450000','#300000']
		args.QPE_cmap = ['#FFFFFF','#D2D2FF','#AAAAFF','#8282FF','#6A6AFF','#4242FF','#1A1AFF','#000090','#000050','#000030']
		args.QPF_cmap = ['#FFFFFF','#D2D2FF','#AAAAFF','#8282FF','#6A6AFF','#4242FF','#1A1AFF','#000090','#000050','#000030']

		args.input_channels = (not args.flow_enable)*1 + args.flow_enable*args.flow_nframes + args.input_with_grid*2 + args.input_with_height*1

		args.TW_map_file = make_path(os.path.join('07_gis_data','03_TW_shapefile','gadm36_TWN_2'), args.working_folder)

		# make folders' path
		args.result_folder = os.path.join(args.result_folder, args.model.upper())
		args.params_folder = os.path.join(args.params_folder, args.model.upper())

		size = '{}X{}'.format(args.F_shape[0], args.F_shape[1])

		if len(args.weather_list) != 0:
			args.result_folder = os.path.join(args.result_folder, '_weather')
			args.params_folder = os.path.join(args.params_folder, '_weather')

		if args.normalize_input:
			args.result_folder += '_ninput'
			args.params_folder += '_ninput'
		
		if args.normalize_target:
			args.result_folder += '_ntarget'
			args.params_folder += '_ntarget'
		
		if args.input_with_grid:
			args.result_folder += '_grid'
			args.params_folder += '_grid'

		
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

		if args.loc_catcher:
			args.result_folder += '_chatchloc'
			args.params_folder += '_chatchloc'

		# denoised data as inputs
		if args.denoise_RAD:
			args.radar_wrangled_data_folder += '_denoise'

		createfolder(args.result_folder)
		createfolder(args.params_folder)

		return args