#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 19:38:22 2022

@author: lds
"""

import config
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


class Profile_Dataset(Dataset):
    def __init__(self, path, dim_input):
        super().__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.dataset = np.load(path)

        self.dim_input = dim_input
        self.dataset = torch.from_numpy(self.dataset).float().to(self.device)
        
    def __getitem__(self, index):
        model_input = self.dataset[index, 0:self.dim_input*3].reshape([3, self.dim_input])
        load_gt = self.dataset[index, self.dim_input*3:].reshape([1, self.dim_input])
        mask = self.dataset[index, self.dim_input*1:self.dim_input*2].bool().reshape([1, self.dim_input])

        return model_input, mask, load_gt

    def __len__(self):
        return len(self.dataset)

trainset = Profile_Dataset(path = config.TRAIN_SET_PTH, dim_input = config.DIM_INPUT)
trainloader = DataLoader(trainset,batch_size=config.BATCH_SIZE, num_workers=0, shuffle=True)
trainloader_eval = DataLoader(trainset,batch_size=1, num_workers=0, shuffle=False)

devset = Profile_Dataset(path = config.DEV_SET_PTH, dim_input = config.DIM_INPUT)
devloader = DataLoader(devset,batch_size=config.BATCH_SIZE, num_workers=0, shuffle=True)
devloader_eval = DataLoader(devset,batch_size=1, num_workers=0, shuffle=False)

testset = Profile_Dataset(path = config.TEST_SET_PTH, dim_input = config.DIM_INPUT)
testloader = DataLoader(testset, batch_size=1, num_workers=0, shuffle=False)