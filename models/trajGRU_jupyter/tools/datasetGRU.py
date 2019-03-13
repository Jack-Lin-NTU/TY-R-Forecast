import os

import datetime as dt

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from .args_tools import args

class TyDataset(Dataset):
    """
    Typhoon dataset

    """

    def __init__(self, ty_list_file, root_dir, train=True, train_num=11, input_frames=5,
                 output_frames=18, with_grid=False, transform=None):
        """
        Args:
            ty_list_file (string): Path of the typhoon list file.
            root_dir (string): Directory with all the files.
            train (boolean): Construct training set or not.
            train_num (int): The event number of training set.
            test_num (int): The event number of testing set.
            input_frames (int, 10-minutes-based): The frames of input data.
            output_frames (int, 10-minutes-based): The frames of output data.
            with_grid (boolean): The option to add gird info into input frames.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        super().__init__()
        self.ty_list = pd.read_excel(ty_list_file,index_col="En name").drop("Ch name",axis=1)
        self.ty_list.index.name = "Typhoon"
        self.root_dir = root_dir
        self.input_frames = input_frames
        self.output_frames = output_frames
        self.transform = transform
        self.with_grid = with_grid
        if with_grid:
            self.gird_x, self.gird_y = np.meshgrid(np.arange(0,args.I_shape[0]),np.arange(0,args.I_shape[0]))
        if train:
            self.events_num = range(0, train_num)
            self.events_list = self.ty_list.index[self.events_num]
        else:
            self.events_num = range(train_num, len(self.ty_list))
            self.events_list = self.ty_list.index[self.events_num]

        tmp = 0
        self.idx_list = pd.DataFrame([],columns=["The starting time", "The ending time",
                                                 "The starting idx","The ending idx"])
        for i in self.events_num:
            frame_s = self.ty_list.iloc[i,0]
            frame_e = self.ty_list.iloc[i,1]-dt.timedelta(minutes=(input_frames+output_frames-1)*10)

            self.total_frames = tmp + int((frame_e-frame_s).days*24*6 + (frame_e-frame_s).seconds/600)+1
            self.idx_list = self.idx_list.append({"The starting time":frame_s,"The ending time":frame_e
                                                  ,"The starting idx":tmp,"The ending idx":self.total_frames-1}
                                                 ,ignore_index=True)
            tmp = self.total_frames

        self.idx_list.index = self.events_list
        self.idx_list.index.name = "Typhoon"

    def __len__(self):
        return self.total_frames

    def print_idx_list(self):
        return self.idx_list

    def print_ty_list(self):
        return self.ty_list

    def __getitem__(self,idx):
        # To identify which event the idx is in.
        assert idx < self.total_frames, 'Index is out of the range of the data!'

        for i in self.idx_list.index:
            if idx > self.idx_list.loc[i,"The ending idx"]:
                continue
            else:
                # determine some indexes
                idx_tmp = idx - self.idx_list.loc[i, "The starting idx"]
                # set typhoon's name
                ty_name = i

                year = str(self.idx_list.loc[i, "The starting time"].year)

                # RAD data(a tensor with shape (input_frames X C X H X W))
                rad_data = []
                for j in range(self.input_frames):
                    file_time=dt.datetime.strftime(self.idx_list.loc[i,"The starting time"]+dt.timedelta(minutes=10*(idx_tmp+j)),format="%Y%m%d%H%M")
                    data_path=os.path.join(self.root_dir,'RAD',(year+'.'+ty_name+"."+file_time+".npz"))
                    data = np.load(data_path)['data'][args.I_y_low:(args.I_y_high+1), args.I_x_left:(args.I_x_right+1)]
                    if self.with_grid:
                        rad_data.append(np.array([data, self.gird_x, self.gird_y]))
                    else:
                        rad_data.append(np.expand_dims(data, axis=0))
                rad_data = np.array(rad_data)

                # QPE data(a tensor with shape (output_frames X H X W))
                qpe_data = []

                for j in range(self.input_frames,self.input_frames+self.output_frames):
                    file_time = dt.datetime.strftime(self.idx_list.loc[i,"The starting time"]+dt.timedelta(minutes=10*(idx_tmp+j)),format="%Y%m%d%H%M")
                    data_path = os.path.join(self.root_dir,'QPE',(year+'.'+ty_name+"."+file_time+".npz"))
                    data = np.load(data_path)['data'][args.F_y_low:args.F_y_high+1,args.F_x_left:args.F_x_right+1]

                    qpe_data.append(data)
                qpe_data = np.array(qpe_data)
                # return the idx of sample
                self.sample = {"RAD": rad_data, "QPE": qpe_data}
                if self.transform:
                    self.sample = self.transform(self.sample)
                return self.sample

class ToTensor(object):
    """Convert ndarrays in samples to Tensors."""
    def __call__(self, sample):
        rad_data, qpe_data = sample['RAD'], sample['QPE']

        # numpy data: x_tsteps X H X W
        # torch data: x_tsteps X H X W
        return {'RAD': torch.from_numpy(rad_data),
                'QPE': torch.from_numpy(qpe_data)}

class Normalize(object):
    """Normalize samples"""
    def __init__(self, mean, std, with_grid=False, normalize_target=False):
        self.mean = mean
        self.std = std
        self.normalize_target = normalize_target
        self.with_grid = with_grid
    def __call__(self, sample):
        rad_data, qpe_data = sample['RAD'], sample['QPE']
        if not self.normalize_target:
            if type(self.mean) and type(self.std) == list:
                for i in range(len(self.mean)):
                    if i < rad_data.shape[0]:
                        if self.with_grid:
                            rad_data[i,0,:,:] = (rad_data[i,0,:,:] - self.mean[i]) / self.std[i]
                            rad_data[i,1,:,:] = rad_data[i,1,:,:]/rad_data.shape[2]
                            rad_data[i,2,:,:] = rad_data[i,2,:,:]/rad_data.shape[2]
                        else:
                            rad_data[i] = (rad_data[i] - self.mean[i]) / self.std[i]
                    else:
                        break
        else:
            if type(self.mean) and type(self.std) == list:
                for i in range(len(self.mean)):
                    if i < rad_data.shape[0]:
                        if self.with_grid:
                            rad_data[i,0,:,:] = (rad_data[i,0,:,:] - self.mean[i]) / self.std[i]
                            rad_data[i,1,:,:] = rad_data[i,1,:,:]/rad_data.shape[2]
                            rad_data[i,2,:,:] = rad_data[i,2,:,:]/rad_data.shape[2]
                        else:
                            rad_data[i] = (rad_data[i] - self.mean[i]) / self.std[i]
                    else:
                        qpe_data[i-rad_data.shape[0]] = (qpe_data[i-rad_data.shape[0]] - self.mean[i]) / self.std[i]

        # numpy data: x_tsteps X H X W
        # torch data: x_tsteps X H X W
        return {'RAD': rad_data,
                'QPE': qpe_data}


# if __name__ == "__main__":
#     # test dataloader
#     input_frames = 5
#     output_frames = 18
#     mean = [12.834] * input_frames
#     mean += [12.834] * output_frames
#     std = [14.14] * input_frames
#     std += [14.14] * output_frames

#     train_dataset = TyDataset(ty_list_file=args.ty_list_file,
#                               root_dir=args.root_dir,
#                               input_frames=5,
#                               output_frames=18,
#                               train=True,
#                               with_grid=True,
#                               transform = Compose([ToTensor(),Normalize(mean,std,with_grid=True)]))
#     # print(train_dataset.print_idx_list())
#     ## torch size = [5,3,180,180]
#     # print(train_dataset[0]["RAD"][0][0].sum())
#     print(train_dataset[0]["RAD"].shape)

#     train_dataset = TyDataset(ty_list_file=args.ty_list_file,
#                               root_dir=args.root_dir,
#                               input_frames=5,
#                               output_frames=18,
#                               train=True,
#                               transform = Compose([ToTensor(),Normalize(mean,std,normalize_target=True)]))
#     # print(train_dataset.print_idx_list())
#     ## torch size = [5,1,180,180]
#     print(train_dataset[0]["RAD"][0][0].sum())
#     print(train_dataset[0]["RAD"].shape)

#     train_dataset = TyDataset(ty_list_file=args.ty_list_file,
#                               root_dir=args.root_dir,
#                               input_frames=5,
#                               output_frames=18,
#                               train=True,
#                               transform = Compose([ToTensor()]))
#     # print(train_dataset.print_idx_list())
#     # print(train_dataset[0]["RAD"].sum())
#     # print(train_dataset[0]["RAD"].shape)
