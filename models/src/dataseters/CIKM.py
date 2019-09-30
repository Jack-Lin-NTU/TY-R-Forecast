import os
import numpy as np
import torch as t
from torch.utils.data import Dataset

class CIKMDataset(Dataset):
    ''' CIKM DATASET. '''
    def __init__(self, train=True, test=False, CIKMfolder='~/ssd/01_ty_research/08_CIKM'):
        super(CIKMDataset, self).__init__()
        CIKMfolder = os.path.expanduser(CIKMfolder)
        if train:
            self.filepath = [os.path.join(CIKMfolder, 'train', x) \
                for x in sorted(os.listdir(os.path.join(CIKMfolder, 'train')))]
        else:
            self.filepath = [os.path.join(CIKMfolder, 'testA', x) \
                for x in sorted(os.listdir(os.path.join(CIKMfolder, 'testA')))]
        if test:
            self.filepath = [os.path.join(CIKMfolder, 'testB', x) \
                for x in sorted(os.listdir(os.path.join(CIKMfolder, 'testB')))]
        
    def __len__(self):
        return len(self.filepath)

    def __getitem__(self, idx):
        assert idx < self.__len__(), 'Index is out of the range of all data!'
        data = np.load(self.filepath[idx])[:,3]
        self.sample = {'inputs': data[:5], 'targets': data[5:]}
        return self.sample

class ToTensor():
    def __call__(self, sample):
        return {'inputs': torch.from_numpy(sample['inputs'].unsqueeze(1)),
                'targets': torch.from_numpy(sample['targets'].unsqueeze(1))}

class Normalize():
    pass