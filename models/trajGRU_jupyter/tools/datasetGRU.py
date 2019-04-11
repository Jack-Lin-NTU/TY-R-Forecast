import os
import datetime as dt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from .args_tools import args

class TyDataset(Dataset):
    '''
    Typhoon dataset
    '''
    def __init__(self, ty_list=args.ty_list, radar_wrangled_data_folder=args.radar_wrangled_data_folder,
                 weather_wrangled_data_folder=args.weather_wrangled_data_folder, 
                 ty_info_wrangled_data_folder=args.ty_info_wrangled_data_folder,
                 weather_list=args.weather_list, input_with_QPE=args.input_with_QPE, 
                 input_with_grid=args.input_with_grid, train=True, train_num=None, 
                 input_frames=args.input_frames, target_frames=args.target_frames,  transform=None):
        '''
        Args:
            ty_list (string): Path of the typhoon list file.
            radar_wrangled_data_folder (string): Folder of radar wrangled data.
            weather_wrangled_data_folder (string): Folder of weather wrangled data.
            ty_info_wrangled_data_folder (string): Folder of ty-info wrangled data.
            weather_list (list): A list of weather infos.
            train (boolean): Construct training set or not.
            train_num (int): The event number of training set.
            test_num (int): The event number of testing set.
            input_frames (int, 10-minutes-based): The frames of input data.
            target_frames (int, 10-minutes-based): The frames of output data.
            input_with_grid (boolean): The option to add gird info into input frames.
            transform (callable, optional): Optional transform to be applied on a sample.
        '''
        super().__init__()
        ty_list = pd.read_csv(ty_list, index_col='En name').drop('Ch name', axis=1)
        ty_list['Time of issuing'] = pd.to_datetime(ty_list['Time of issuing'])
        ty_list['Time of canceling'] = pd.to_datetime(ty_list['Time of canceling'])
        ty_list.index.name = 'Typhoon'
        
        self.ty_list = ty_list
        self.radar_wrangled_data_folder = radar_wrangled_data_folder
        self.weather_wrangled_data_folder = weather_wrangled_data_folder
        self.ty_info_wrangled_data_folder = ty_info_wrangled_data_folder
        self.weather_list = weather_list
        self.input_with_grid = input_with_grid
        self.input_with_QPE = input_with_QPE
        
        self.input_frames = input_frames
        self.target_frames = target_frames
        self.transform = transform
        
        if input_with_grid:
            self.gird_x, self.gird_y = np.meshgrid(np.arange(0, args.I_shape[0]), np.arange(0, args.I_shape[0]))
        
        if train:
            if train_num is None:
                self.events_num = np.arange(0, int(len(self.ty_list)/5*4))
            else:
                assert train_num <= len(self.ty_list), 'The train_num shoud be less than total number of ty events.'
                self.events_num = np.arange(0, train_num)
            self.events_list = self.ty_list.index[self.events_num]
        else:
            if train_num is None:
                self.events_num = np.arange(int(len(self.ty_list)/5*4), len(self.ty_list))
            else:
                assert train_num <= len(self.ty_list), 'The train_num shoud be less than total number of ty events.'
                self.events_num = np.arange(train_num, len(self.ty_list))
            self.events_list = self.ty_list.index[self.events_num]

        tmp = 0
        self.idx_list = pd.DataFrame([], columns=['The starting time', 'The ending time', 'The starting idx', 'The ending idx'], 
                                     index=self.events_list)
    
        for i in self.idx_list.index:
            frame_s = self.ty_list.loc[i, 'Time of issuing']
            frame_e = self.ty_list.loc[i, 'Time of canceling'] - dt.timedelta(minutes=(input_frames+target_frames-1)*10)
            
            self.total_frames = tmp + int((frame_e-frame_s).days*24*6 + (frame_e-frame_s).seconds/600) + 1
            self.idx_list.loc[i,:] = [frame_s, frame_e, tmp, self.total_frames-1]
            tmp = self.total_frames
        
        self.idx_list.index = self.events_list
        self.idx_list.index.name = 'Typhoon'

    def __len__(self):
        return self.total_frames

    def print_idx_list(self):
        return self.idx_list

    def print_ty_list(self):
        return self.ty_list

    def __getitem__(self, idx):
        # To identify which event the idx is in.
        assert idx < self.total_frames, 'Index is out of the range of the data!'

        for i in self.idx_list.index:
            if idx > self.idx_list.loc[i, 'The ending idx']:
                continue
            else:
                # determine some indexes
                idx_tmp = idx - self.idx_list.loc[i, 'The starting idx']
                # set typhoon's name
                ty_name = i

                year = str(self.idx_list.loc[i, 'The starting time'].year)

                # Input data(a tensor with shape (input_frames X C X H X W))
                input_data = []
                for j in range(self.input_frames):
                    tmp_data = []
                    # Radar
                    file_time = dt.datetime.strftime(self.idx_list.loc[i,'The starting time']+dt.timedelta(minutes=10*(idx_tmp+j)), format='%Y%m%d%H%M')
                    data_path = os.path.join(self.radar_wrangled_data_folder, 'RAD', year+'.'+ty_name+'.'+file_time+'.pkl')
                    tmp_data.append(pd.read_pickle(data_path, compression=args.compression).loc[args.I_y[1]:args.I_y[0], args.I_x[0]:args.I_x[1]].to_numpy())
                    
                    if self.input_with_QPE:
                        data_path = os.path.join(self.radar_wrangled_data_folder, 'QPE', year+'.'+ty_name+'.'+file_time+'.pkl')
                        tmp_data.append(pd.read_pickle(data_path, compression=args.compression).loc[args.I_y[1]:args.I_y[0], args.I_x[0]:args.I_x[1]].to_numpy())
                    
                    for k in self.weather_list:
                        data_path = os.path.join(self.weather_wrangled_data_folder, k, (year+'.'+ty_name+'.'+file_time+'.pkl'))
                        tmp_data.append(pd.read_pickle(data_path, compression=args.compression).loc[args.I_y[1]:args.I_y[0], args.I_x[0]:args.I_x[1]].to_numpy())
                    
                    if self.input_with_grid:
                        input_data.append(np.array(tmp_data+[self.gird_x, self.gird_y]))
                    else:
                        input_data.append(np.array(tmp_data))
                input_data = np.array(input_data)

                # QPE data(a tensor with shape (target_frames X H X W))
                target_data = []

                for j in range(self.input_frames, self.input_frames+self.target_frames):
                    file_time = dt.datetime.strftime(self.idx_list.loc[i,'The starting time']+dt.timedelta(minutes=10*(idx_tmp+j)),format='%Y%m%d%H%M')
                    data_path = os.path.join(self.radar_wrangled_data_folder, 'QPE', year+'.'+ty_name+'.'+file_time+'.pkl')
                    target_data.append(pd.read_pickle(data_path, compression=args.compression).loc[args.F_y[1]:args.F_y[0], args.F_x[0]:args.F_x[1]].to_numpy())
                target_data = np.array(target_data)
                # return the idx of sample
                self.sample = {'input': input_data, 'target': target_data}
                
                if self.transform:
                    self.sample = self.transform(self.sample)
                return self.sample

class ToTensor(object):
    '''Convert ndarrays in samples to Tensors.'''
    def __call__(self, sample):
        # numpy data: x_tsteps X H X W
        # torch data: x_tsteps X H X W
        return {'input': torch.from_numpy(sample['input']),
                'target': torch.from_numpy(sample['target'])}

class Normalize(object):
    '''
    Normalize samples
    '''
    def __init__(self, max_values, min_values, input_with_QPE=args.input_with_QPE, input_with_grid=args.input_with_grid, normalize_target=args.normalize_target):
        assert type(max_values) == pd.Series or list, 'max_values is a not pd.series or list.'
        assert type(min_values) == pd.Series or list, 'min_values is a not pd.series or list.'
        self.max_values = max_values
        self.min_values = min_values
        self.normalize_target = normalize_target
        self.input_with_grid = input_with_grid
        self.input_with_QPE = input_with_QPE
        
    def __call__(self, sample):
        input_data, target_data = sample['input'], sample['target']
        
        index = 0
        input_data[:,index,:,:] = (input_data[:,index,:,:] - self.min_values['RAD']) / (self.max_values['RAD'] - self.min_values['RAD'])
        
        if self.input_with_QPE:
            index += 1
            input_data[:,index,:,:] = (input_data[:,index,:,:] - self.min_values['QPE']) / (self.max_values['QPE'] - self.min_values['QPE'])

        for idx, value in enumerate(args.weather_list):
            index += 1
            input_data[:,index,:,:] = (input_data[:,index,:,:] - self.min_values[value]) / (self.max_values[value] - self.min_values[value])
        
        if self.input_with_grid:    
            index += 1
            input_data[:,index,:,:] = input_data[:,index,:,:] / args.I_shape[0]
            index += 1
            input_data[:,index,:,:] = input_data[:,index,:,:] / args.I_shape[1]
        
        if self.normalize_target:
            target_data = (target_data - self.min_values['QPE']) / (self.max_values['QPE'] - self.min_values['QPE'])

        # numpy data: x_tsteps X H X W
        # torch data: x_tsteps X H X W
        return {'input': input_data,
                'target': target_data}
    
    
if __name__ == '__main__':
    transform = Compose([ToTensor(), Normalize(max_values=args.max_values, min_values=args.min_values)])
    a = TyDataset(ty_list = args.ty_list,
                  input_frames = args.input_frames,
                  target_frames = args.target_frames,
                  train = True,
                  input_with_QPE = args.input_with_QPE,
                  input_with_grid = args.input_with_grid,
                  transform=transform)
    
    print(a[0]['input'])