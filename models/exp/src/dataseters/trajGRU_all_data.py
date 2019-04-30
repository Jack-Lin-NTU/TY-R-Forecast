import os
import datetime as dt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class TyDataset(Dataset):
    '''
    Typhoon dataset
    '''
    def __init__(self, args=None, train=True, train_num=None, transform=None):
        '''
        Args:
            train (boolean): Construct training set or not.
            train_num (int): The event number of training set.
            transform (callable, optional): Optional transform to be applied on a sample.
        '''
        super().__init__()
        if args is not None:
            ty_list = pd.read_csv(args.ty_list, index_col='En name').drop('Ch name', axis=1)
            ty_list['Time of issuing'] = pd.to_datetime(ty_list['Time of issuing'])
            ty_list['Time of canceling'] = pd.to_datetime(ty_list['Time of canceling'])
            ty_list.index.name = 'Typhoon'
            
            self.ty_list = ty_list
            self.radar_wrangled_data_folder = args.radar_wrangled_data_folder
            self.weather_wrangled_data_folder = args.weather_wrangled_data_folder
            self.ty_info_wrangled_data_folder = args.ty_info_wrangled_data_folder
            self.weather_list = args.weather_list
            self.input_with_grid = args.input_with_grid
            self.input_with_QPE = args.input_with_QPE
            self.input_frames = args.input_frames
            self.target_frames = args.target_frames
            self.input_channels = args.input_channels
            self.I_x = args.I_x
            self.I_y = args.I_y
            self.I_shape = args.I_shape
            self.F_x = args.F_x
            self.F_y = args.F_y
            self.F_shape = args.F_shape

        self.transform = transform
        
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
            frame_e = self.ty_list.loc[i, 'Time of canceling'] - dt.timedelta(minutes=(self.input_frames+self.target_frames-1)*10)
            
            self.total_frames = tmp + int((frame_e-frame_s).days*24*6 + (frame_e-frame_s).seconds/600) + 1
            self.idx_list.loc[i,:] = [frame_s, frame_e, tmp, self.total_frames-1]
            tmp = self.total_frames
        
        self.idx_list.index = self.events_list
        self.idx_list.index.name = 'Typhoon'

        if args.load_all_data:
            self.rad_all_data = pd.Series([])
            self.qpe_all_data = pd.Series([])

            # for i in range(len(self.idx_list)):
            for i in range(len(self.idx_list)):
                year = str(self.idx_list.iloc[i,0].year)
                for j in range(self.idx_list.iloc[i,-1]-self.idx_list.iloc[i,-2]+self.input_frames+self.target_frames):
                    filename = year + '.' + self.idx_list.index[i] + '.' + dt.datetime.strftime(self.idx_list.iloc[i,0]+dt.timedelta(minutes=10*j),format='%Y%m%d%H%M') + '.pkl'
                    self.rad_all_data[filename[:-4]] = pd.read_pickle(os.path.join(args.radar_wrangled_data_folder, 'RAD', filename),compression=args.compression).loc[self.I_y[0]:self.I_y[1], self.I_x[0]:self.I_x[1]]
                    self.qpe_all_data[filename[:-4]] = pd.read_pickle(os.path.join(args.radar_wrangled_data_folder, 'QPE', filename),compression=args.compression).loc[self.I_y[0]:self.I_y[1], self.I_x[0]:self.I_x[1]]

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
                ty_name = i
                year = str(self.idx_list.loc[i, 'The starting time'].year)
                # set default input_data (input_frames X channels X H X W)
                input_data = np.zeros((self.input_frames, self.input_channels, self.I_shape[1], self.I_shape[0]))
                # Input data(a tensor with shape (input_frames X C X H X W))
                for j in range(self.input_frames):
                    # Radar
                    tmp = 0
                    file_time = dt.datetime.strftime(self.idx_list.loc[i,'The starting time']+dt.timedelta(minutes=10*(idx_tmp+j)), format='%Y%m%d%H%M')
                    input_data[j,tmp,:,:] = self.rad_all_data[year+'.'+ty_name+'.'+file_time].to_numpy()
                    tmp += 1
                    # QPE
                    if self.input_with_QPE:
                        input_data[j,tmp,:,:] = self.qpe_all_data[year+'.'+ty_name+'.'+file_time].to_numpy()
                        tmp += 1
                    
                    if self.input_with_grid:
                        gird_x, gird_y = np.meshgrid(np.arange(0, self.I_shape[0]), np.arange(0, self.I_shape[1]))
                        input_data[j,tmp,:,:] = gird_x
                        tmp += 1
                        input_data[j,tmp,:,:] = gird_y
                        tmp += 1 
                
                # set default target_data (target_frames X channels X H X W)
                target_data = np.zeros((self.target_frames, self.F_shape[1], self.F_shape[0]))
                idx_tmp += self.input_frames
                for j in range(self.target_frames):
                    file_time = dt.datetime.strftime(self.idx_list.loc[i,'The starting time']+dt.timedelta(minutes=10*(idx_tmp+j)), format='%Y%m%d%H%M')
                    target_data[j,:,:] = self.qpe_all_data[year+'.'+ty_name+'.'+file_time].loc[self.F_y[0]:self.F_y[1], self.F_x[0]:self.F_x[1]].to_numpy()

                # return the idx of sample
                self.sample = {'inputs': input_data, 'targets': target_data}
                
                if self.transform:
                    self.sample = self.transform(self.sample)
                return self.sample

class ToTensor(object):
    '''Convert ndarrays in samples to Tensors.'''
    def __call__(self, sample):
        # numpy data: x_tsteps X H X W
        # torch data: x_tsteps X H X W
        return {'inputs': torch.from_numpy(sample['inputs']), 'targets': torch.from_numpy(sample['targets'])}

class Normalize(object):
    '''
    Normalize samples
    '''
    def __init__(self, args):
        assert type(args.max_values) == pd.Series or list, 'max_values is a not pd.series or list.'
        assert type(args.min_values) == pd.Series or list, 'min_values is a not pd.series or list.'
        self.max_values = args.max_values
        self.min_values = args.min_values
        self.normalize_target = args.normalize_target
        self.input_with_grid = args.input_with_grid
        self.input_with_QPE = args.input_with_QPE
        self.weather_list = args.weather_list
        self.I_shape = args.I_shape
        
    def __call__(self, sample):
        input_data, target_data = sample['inputs'], sample['targets']
        
        index = 0
        input_data[:,index,:,:] = (input_data[:,index, :, :] - self.min_values['RAD']) / (self.max_values['RAD'] - self.min_values['RAD'])
        
        if self.input_with_QPE:
            index += 1
            input_data[:,index,:,:] = (input_data[:,index,:,:] - self.min_values['QPE']) / (self.max_values['QPE'] - self.min_values['QPE'])

        for idx, value in enumerate(self.weather_list):
            index += 1
            input_data[:,index,:,:] = (input_data[:,index,:,:] - self.min_values[value]) / (self.max_values[value] - self.min_values[value])
        
        if self.input_with_grid:    
            index += 1
            input_data[:,index,:,:] = input_data[:,index,:,:] / self.I_shape[0]
            index += 1
            input_data[:,index,:,:] = input_data[:,index,:,:] / self.I_shape[1]
        
        if self.normalize_target:
            target_data = (target_data - self.min_values['QPE']) / (self.max_values['QPE'] - self.min_values['QPE'])

        # numpy data: x_tsteps X H X W
        # torch data: x_tsteps X H X W
        return {'inputs': input_data, 'targets': target_data}
