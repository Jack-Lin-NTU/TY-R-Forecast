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
    def __init__(self, args, train=True, train_num=None, transform=None):
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
            self.O_shape = args.O_shape
            self.compression = args.compression
            self.target_RAD = args.target_RAD

        self.transform = transform
        rand_tys = np.random.choice(len(ty_list), len(ty_list), replace=False)
        if train:
            if train_num is None:
                self.events_num = rand_tys[0:int(len(self.ty_list)/4*3)]
            else:
                assert train_num <= len(self.ty_list), 'The train_num shoud be less than total number of ty events.'
                self.events_num = rand_tys[0:train_num]
            self.events_list = self.ty_list.index[self.events_num]
        else:
            if train_num is None:
                self.events_num = rand_tys[int(len(self.ty_list)/4*3):]
            else:
                assert train_num <= len(self.ty_list), 'The train_num shoud be less than total number of ty events.'
                self.events_num = rand_tys[train_num:]
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
                # print('idx:',idx)
                # determine some indexes
                idx_tmp = idx - self.idx_list.loc[i, 'The starting idx']
                # set typhoon's name
                ty_name = i
                year = str(self.idx_list.loc[i, 'The starting time'].year)

                # Input data(a tensor with shape (input_frames X C X H X W)) (0-5)
                input_data = np.zeros((self.input_frames, self.input_channels, self.I_shape[1], self.I_shape[0]))
                # Radar Map(a tensor with shape (C X H X W)) last input image
                radar_map = np.zeros((self.input_channels, self.O_shape[1], self.O_shape[0]))

                for j in range(self.input_frames):
                    # Radar
                    tmp = 0
                    file_time = dt.datetime.strftime(self.idx_list.loc[i,'The starting time']+dt.timedelta(minutes=10*(idx_tmp+j)), format='%Y%m%d%H%M')
                    data_path = os.path.join(self.radar_wrangled_data_folder, 'RAD', year+'.'+ty_name+'.'+file_time+'.pkl')
                    input_data[j,tmp,:,:] = pd.read_pickle(data_path, compression=self.compression).loc[self.I_y[0]:self.I_y[1], self.I_x[0]:self.I_x[1]].to_numpy()
                    tmp += 1
                    
                    if self.input_with_QPE:
                        data_path = os.path.join(self.radar_wrangled_data_folder, 'QPE', year+'.'+ty_name+'.'+file_time+'.pkl')
                        input_data[j,tmp,:,:] = pd.read_pickle(data_path, compression=self.compression).loc[self.I_y[0]:self.I_y[1], self.I_x[0]:self.I_x[1]].to_numpy()
                        tmp += 1

                    # for k in self.weather_list:
                    #     data_path = os.path.join(self.weather_wrangled_data_folder, k, (year+'.'+ty_name+'.'+file_time+'.pkl'))
                    #     input_data[j,tmp,:,:] = pd.read_pickle(data_path, compression=self.compression).loc[self.I_y[1]:self.I_y[0],self.I_x[0]:self.I_x[1]].to_numpy()[np.newaxis,np.newaxis,:,:]
                    #     tmp += 1

                    if self.input_with_grid:
                        gird_x, gird_y = np.meshgrid(np.arange(0, self.I_shape[0]), np.arange(0, self.I_shape[1]))
                        input_data[j,tmp,:,:] = gird_x
                        tmp += 1
                        input_data[j,tmp,:,:] = gird_y
                        tmp += 1

                    # RADARMAP (last input image)
                    if j == self.input_frames-1:
                        tmp = 0
                        file_time = dt.datetime.strftime(self.idx_list.loc[i,'The starting time']+dt.timedelta(minutes=10*(idx_tmp+j)), format='%Y%m%d%H%M')

                        data_path = os.path.join(self.radar_wrangled_data_folder, 'RAD', year+'.'+ty_name+'.'+file_time+'.pkl')
                        radar_map[tmp,:,:] = pd.read_pickle(data_path, compression=self.compression).to_numpy()

                        if self.input_with_QPE:
                                data_path = os.path.join(self.radar_wrangled_data_folder, 'QPE', year+'.'+ty_name+'.'+file_time+'.pkl')
                                radar_map[tmp,:,:] = pd.read_pickle(data_path, compression=self.compression).to_numpy()
                                tmp += 1

                        # for k in self.weather_list:
                        #     data_path = os.path.join(self.weather_wrangled_data_folder, k, (year+'.'+ty_name+'.'+file_time+'.pkl'))
                        #     input_data[tmp,:,:] = pd.read_pickle(data_path, compression=self.compression).to_numpy()[np.newaxis,np.newaxis,:,:]
                        #     tmp += 1

                        if self.input_with_grid:
                            gird_x, gird_y = np.meshgrid(np.arange(0, self.O_shape[0]), np.arange(0, self.O_shape[1]))
                            radar_map[tmp,:,:] = gird_x
                            tmp += 1
                            radar_map[tmp,:,:] = gird_y
                            tmp += 1
                
                # update index of time
                idx_tmp += self.input_frames
                # TY infos (6-24)
                data_path = os.path.join(self.ty_info_wrangled_data_folder, year+'.'+ty_name+'.csv')
                ty_infos = pd.read_csv(data_path)
                ty_infos.Time = pd.to_datetime(ty_infos.Time)
                ty_infos = ty_infos.set_index('Time')

                file_time1 = dt.datetime.strftime(self.idx_list.loc[i,'The starting time']+dt.timedelta(minutes=10*idx_tmp), format='%Y%m%d%H%M')
                file_time2 = dt.datetime.strftime(self.idx_list.loc[i,'The starting time']+dt.timedelta(minutes=10*(idx_tmp+self.target_frames-1)), format='%Y%m%d%H%M')

                ty_infos = ty_infos.loc[file_time1:file_time2,:].to_numpy()
                
                # QPE data(a tensor with shape (target_frames X H X W)) (6-24)
                target_data = np.zeros((self.target_frames, self.F_shape[1], self.F_shape[0]))
                if self.target_RAD:
                    data_type = 'RAD'
                else:
                    data_type = 'QPE'
                for j in range(self.target_frames):
                    file_time = dt.datetime.strftime(self.idx_list.loc[i,'The starting time']+dt.timedelta(minutes=10*(idx_tmp+j)), format='%Y%m%d%H%M')
                    data_path = os.path.join(self.radar_wrangled_data_folder, data_type, year+'.'+ty_name+'.'+file_time+'.pkl')
                    target_data[j,:,:] = pd.read_pickle(data_path, compression=self.compression).loc[self.F_y[0]:self.F_y[1], self.F_x[0]:self.F_x[1]].to_numpy()

                if self.target_RAD:
                    target_data = 

                # the start time of prediction 
                pre_time = dt.datetime.strftime(self.idx_list.loc[i,'The starting time']+dt.timedelta(minutes=10*(idx_tmp)), format='%Y%m%d%H%M')
                # return the idx of sample
                self.sample = {'inputs': input_data, 'targets': target_data, 'ty_infos': ty_infos, 'radar_map': radar_map, 'time': pre_time}
                
                if self.transform:
                    self.sample = self.transform(self.sample)
                return self.sample

class ToTensor(object):
    '''Convert ndarrays in samples to Tensors.'''
    def __call__(self, sample):
        # numpy data: x_tsteps X H X W
        # torch data: x_tsteps X H X W
        return {'inputs': torch.from_numpy(sample['inputs']),
                'targets': torch.from_numpy(sample['targets']), 
                'ty_infos': torch.from_numpy(sample['ty_infos']),
                'radar_map': torch.from_numpy(sample['radar_map']),
                'time': sample['time']}

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
        input_data, target_data, ty_infos, radar_map = sample['inputs'], sample['targets'], sample['ty_infos'], sample['radar_map'] 
        
        # normalize inputs
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
        
        # normalize targets
        if self.normalize_target:
            target_data = (target_data - self.min_values['QPE']) / (self.max_values['QPE'] - self.min_values['QPE'])
        
        # normalize radar map
        index = 0
        radar_map[index,:,:] = (radar_map[index, :, :] - self.min_values['RAD']) / (self.max_values['RAD'] - self.min_values['RAD'])

        if self.input_with_QPE:
            index += 1
            radar_map[index,:,:] = (radar_map[index,:,:] - self.min_values['QPE']) / (self.max_values['QPE'] - self.min_values['QPE'])

        for idx, value in enumerate(self.weather_list):
            index += 1
            radar_map[index,:,:] = (radar_map[index,:,:] - self.min_values[value]) / (self.max_values[value] - self.min_values[value])
        
        if self.input_with_grid:    
            index += 1
            radar_map[index,:,:] = radar_map[index,:,:] / self.I_shape[0]
            index += 1
            radar_map[index,:,:] = radar_map[index,:,:] / self.I_shape[1]
        
        # normalize ty info
        min_values = torch.from_numpy(self.min_values.loc['Lat':].to_numpy())
        max_values = torch.from_numpy(self.max_values.loc['Lat':].to_numpy())

        ty_infos = (ty_infos - min_values) / ( max_values - min_values)

        # numpy data: x_tsteps X H X W
        # torch data: x_tsteps X H X W
        return {'inputs': input_data, 'targets': target_data, 'ty_infos': ty_infos, 'radar_map': radar_map, 'time': sample['time']}
