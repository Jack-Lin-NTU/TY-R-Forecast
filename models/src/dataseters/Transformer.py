import os
import datetime as dt
from easydict import EasyDict as edict
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class TyDataset(Dataset):
    '''
    Typhoon dataset
    '''
    def __init__(self, train=True, args=None, transform=None):
        '''
        Args:
            train(boolean): Return training dataset or validating dataset.
            args(easydict): An edict for saving 
            transform (callable, optional): Optional transform to be applied on a sample.
        '''
        super(TyDataset, self).__init__()

        if args is not None:
            ty_list = pd.read_csv(args.ty_list, index_col='En name').drop('Ch name', axis=1)
            ty_list['Time of issuing'] = pd.to_datetime(ty_list['Time of issuing'])
            ty_list['Time of canceling'] = pd.to_datetime(ty_list['Time of canceling'])
            ty_list.index.name = 'Typhoon'
            
            self.ty_list = ty_list
            self.radar_folder = args.radar_folder
            self.radar_wrangled_data_folder = args.radar_wrangled_data_folder
            self.weather_wrangled_data_folder = args.weather_wrangled_data_folder
            self.ty_info_wrangled_data_folder = args.ty_info_wrangled_data_folder
            self.weather_list = args.weather_list
            self.input_with_grid = args.input_with_grid
            self.I_nframes = args.I_nframes
            self.F_nframes = args.F_nframes
            self.input_channels = args.input_channels
            self.I_x = args.I_x
            self.I_y = args.I_y
            self.I_shape = args.I_shape
            self.F_x = args.F_x
            self.F_y = args.F_y
            self.F_shape = args.F_shape
            self.O_shape = args.O_shape
            self.compression = args.compression
            self.loc_catcher = args.loc_catcher
            self.value_dtype = args.value_dtype
            train_num = args.train_num

        self.transform = transform

        # set random seed
        np.random.seed(args.random_seed)
        randon_events = np.random.choice(len(ty_list), len(ty_list), replace=False)
        
        if train:
            if train_num is None:
                events_num = randon_events[0:int(len(ty_list)/4*3)]
            else:
                assert train_num <= len(ty_list), 'The train_num shoud be less than total number of ty events.'
                events_num = randon_events[0:train_num]
            self.ty_list = self.ty_list.iloc[events_num]
        else:
            if train_num is None:
                events_num = randon_events[int(len(ty_list)/4*3):]
            else:
                assert train_num <= len(ty_list), 'The train_num shoud be less than total number of ty events.'
                events_num = randon_events[train_num:]
            self.ty_list = self.ty_list.iloc[events_num]
            
        frame_s = (self.ty_list.loc[:, 'Time of issuing'])
        frame_e = (self.ty_list.loc[:, 'Time of canceling'] - dt.timedelta(minutes=(self.I_nframes+self.F_nframes-1)*10))
        events_windows = ((frame_e-frame_s).apply(lambda x:x.days)*24*6 + (frame_e-frame_s).apply(lambda x:x.seconds)/600 + 1).astype(int)
        last_idx = np.cumsum(events_windows) - 1
        frist_idx = np.cumsum(events_windows) - events_windows
        self.idx_df = pd.concat([frame_s, frame_e, frist_idx, last_idx], axis=1)
        self.idx_df.columns = ['The s_time of first window', 'The s_time of last window', 'The frist idx', 'The last idx']
        self.idx_df.index = self.ty_list.index
        self.idx_df.index.name = 'Typhoon'

        self.total_frames = last_idx[-1]

    def __len__(self):
        return self.total_frames

    def __getitem__(self, idx):
        # To identify which event the idx is in.
        assert idx < self.total_frames, 'Index is out of the range of the data!'

        for i in self.idx_df.index:
            if idx > self.idx_df.loc[i, 'The last idx']:
                continue
            else:
                # determine some indexes
                tmp_idx = int(idx - self.idx_df.loc[i, 'The frist idx'])

                # save current time
                current_time = dt.datetime.strftime(self.idx_df.loc[i,'The s_time of first window'] + \
                                dt.timedelta(minutes=10*(tmp_idx+self.I_nframes-1)), format='%Y%m%d%H%M')
                
                # typhoon's name
                ty_name = str(self.idx_df.loc[i, 'The s_time of first window'].year)+'.'+i

                # Input data(a tensor with shape (I_nframes X C X H X W)) (0-5)
                input_data = np.zeros((self.I_nframes, self.input_channels, self.I_shape[1], self.I_shape[0]))
                # Target data(a tensor with shape (F_nframes X H X W)) (0-5)
                target_data = np.zeros((self.F_nframes, self.I_shape[1], self.I_shape[0]))
                # Radar Map(a tensor with shape (C X H X W)) last input image
                current_map = np.zeros((self.input_channels, self.O_shape[1], self.O_shape[0]))

                # Read input data
                for j in range(self.I_nframes):
                    c = 0
                    file_time = dt.datetime.strftime(self.idx_df.loc[i,'The s_time of first window'] + \
                                dt.timedelta(minutes=10*(tmp_idx+j)), format='%Y%m%d%H%M')
                    data_path = os.path.join(self.radar_wrangled_data_folder, 'RAD', ty_name+'.'+file_time+'.pkl')
                    input_data[j,c,:,:] = pd.read_pickle(data_path, compression=self.compression).loc[self.I_y[0]:self.I_y[1], self.I_x[0]:self.I_x[1]].to_numpy()
                    c += 1

                    if self.input_with_grid:
                        input_data[j,c,:,:], input_data[j,c+1,:,:] = np.meshgrid(np.arange(0, self.I_shape[0]), np.arange(0, self.I_shape[1]))
                        c += 2

                # current radar map
                c = 0
                file_time = dt.datetime.strftime(self.idx_df.loc[i,'The s_time of first window']+dt.timedelta(minutes=10*(tmp_idx+self.I_nframes-1)), format='%Y%m%d%H%M')
                data_path = os.path.join(self.radar_wrangled_data_folder, 'RAD', ty_name+'.'+file_time+'.pkl')
                current_map[c,:,:] = pd.read_pickle(data_path, compression=self.compression).to_numpy()
                c += 1

                if self.input_with_grid:
                    current_map[c,:,:], current_map[c+1,:,:] = np.meshgrid(np.arange(0, self.O_shape[0]), np.arange(0, self.O_shape[1]))
                    c += 2
                
                # update index of time
                tmp_idx += self.I_nframes

                # TYs-infos (6-24)
                data_path = os.path.join(self.ty_info_wrangled_data_folder, ty_name+'.csv')
                ty_infos = pd.read_csv(data_path)
                ty_infos.Time = pd.to_datetime(ty_infos.Time)
                ty_infos = ty_infos.set_index('Time')

                file_time1 = dt.datetime.strftime(self.idx_df.loc[i,'The s_time of first window'], format='%Y%m%d%H%M')
                file_time2 = dt.datetime.strftime(self.idx_df.loc[i,'The s_time of last window'], format='%Y%m%d%H%M')
                
                ty_infos = ty_infos.loc[file_time1:file_time2,:].to_numpy()
                if self.loc_catcher:
                    ty_infos = ty_infos[:,[0,1,-1]]
                else:
                    ty_infos = ty_infos[:,0:-1]
                
                # Read target data(a tensor with shape (F_nframes X H X W)) (6-24)
                target_data = np.zeros((self.F_nframes, self.F_shape[1], self.F_shape[0]))

                for j in range(self.F_nframes):
                    file_time = dt.datetime.strftime(self.idx_df.loc[i,'The s_time of first window']+dt.timedelta(minutes=10*(tmp_idx+j)), format='%Y%m%d%H%M')
                    data_path = os.path.join(self.radar_wrangled_data_folder, 'RAD', ty_name+'.'+file_time+'.pkl')
                    target_data[j,:,:] = pd.read_pickle(data_path, compression=self.compression).loc[self.F_y[0]:self.F_y[1], self.F_x[0]:self.F_x[1]].to_numpy()

                height = pd.read_pickle(os.path.join(self.radar_folder, 'height.pkl'), compression='bz2').loc[self.I_y[0]:self.I_y[1], self.I_x[0]:self.I_x[1]].to_numpy()

                height = (height-np.min(height))/(np.max(height)-np.min(height))

                self.sample = {'inputs': input_data, 'targets': target_data, 'ty_infos': ty_infos, 'current_map': current_map, 'current_time': current_time, 'height': height}
                
                if self.transform:
                    self.sample = self.transform(self.sample)

                # return the idx of sample
                return self.sample

class ToTensor(object):
    '''Convert ndarrays in samples to Tensors.'''
    def __call__(self, sample):
        return {'inputs': torch.from_numpy(sample['inputs']),
                'height': torch.from_numpy(sample['height']), 
                'targets': torch.from_numpy(sample['targets']), 
                'ty_infos': torch.from_numpy(sample['ty_infos']),
                'current_map': torch.from_numpy(sample['current_map']),
                'current_time': sample['current_time']
                }

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
        self.loc_catcher = args.loc_catcher
        self.I_shape = args.I_shape
        
    def __call__(self, sample):
        input_data, target_data, ty_infos, current_map = sample['inputs'], sample['targets'], sample['ty_infos'], sample['current_map'] 
        # normalize inputs
        index = 0
        input_data[:,index,:,:] = (input_data[:,index, :, :] - self.min_values['RAD']) / (self.max_values['RAD'] - self.min_values['RAD'])
        
        if self.input_with_grid:
            index += 1
            input_data[:,index,:,:] = input_data[:,index,:,:] / self.I_shape[0]
            index += 1
            input_data[:,index,:,:] = input_data[:,index,:,:] / self.I_shape[1]
        
        # normalize targets
        if self.normalize_target:
            target_data = (target_data - self.min_values['RAD']) / (self.max_values['RAD'] - self.min_values['RAD'])

        # normalize current map
        index = 0
        current_map[index,:,:] = (current_map[index, :, :] - self.min_values['RAD']) / (self.max_values['RAD'] - self.min_values['RAD'])
        if self.input_with_grid:    
            index += 1
            current_map[index,:,:] = current_map[index,:,:] / self.I_shape[0]
            index += 1
            current_map[index,:,:] = current_map[index,:,:] / self.I_shape[1]

        # normalize ty info
        min_values = torch.from_numpy(self.min_values.loc['Lat':].to_numpy())
        max_values = torch.from_numpy(self.max_values.loc['Lat':].to_numpy())
        ty_infos = (ty_infos - min_values) / ( max_values - min_values)
        # numpy data: x_tsteps X H X W
        # torch data: x_tsteps X H X W
        return {'inputs': input_data, 'height': height, 'targets': target_data, 'ty_infos': ty_infos, 'current_map': sample['current_map'], 'current_time': sample['current_time']}
