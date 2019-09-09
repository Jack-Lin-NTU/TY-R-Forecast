# This parser file is for juypter notebook.
import os
import numpy as np
import pandas as pd
from easydict import EasyDict as edict
import torch
import torch.optim as optim

def get_parser():
    working_folder = os.path.expanduser('~/ssd/01_ty_research')
    
    parser = edict({
        'ty_list': os.path.join(working_folder, 'ty_list.csv'),
        'radar_folder': os.path.join(working_folder, '01_radar_data'),
        'weather_folder': os.path.join(working_folder, '02_weather_data'),
        'ty_info_folder': os.path.join(working_folder, '03_ty_info'),
        'radar_wrangled_data_folder': os.path.join(working_folder, '01_radar_data', '02_wrangled_files_smooth'),
        'weather_wrangled_data_folder': os.path.join(working_folder, '02_weather_data', '01_wrangled_files'),
        'ty-info-wrangled-data-folder': os.path.join(working_folder, '03_ty_info', '01_wrangled_files'),
        'result_folder': os.path.join(working_folder, '04_results'),
        'params_folder': os.path.join(working_folder, '05_params'),
        'infer_folder': os.path.join(working_folder, '06_inferences'),
        'able_cuda': True,
        'gpu': 0,
        'value_dtype': torch.float32,
        'randome_seed': 1000,
        'train_num': 10,
         'max_epochs': 30,
         'batch_size': 8,
         'lr': 0.0001,
         'lr_scheduler': True,
         'wd': 0.01,
         'clip': True,
         'clip_max_norm': 1,
         'batch-norm': Ture,
         '': 8,
         '': 8,
         '': 8,
         '': 8,
         '': 8,
         '': 8,
         '': 8,
         '': 8,
         '': 8,
         '': 8,
         '': 8,
         '': 8,}
    )