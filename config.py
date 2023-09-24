import torch
import os
from model import GenGIN

DIM_INPUT = 288
DIM_PATCH = 4*12
CH_INPUT = 3 # 0-load(with hole)/1-mask/2-temperature
N_EPOCH = 301
SAVE_PER_EPO = 10
BATCH_SIZE = 16
LR = 3e-4
NF_GEN = 64
NF_DIS = 8
W_P2P = 0.5
W_GAN = 0.02
W_FEA = 0.2
DROPRATE = 0.0
USE_LOCAL_GAN_LOSS = True
TAG = ''

CUDA = True if torch.cuda.is_available() else False
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
Tensor = torch.cuda.FloatTensor if CUDA else torch.FloatTensor

DIR = "../plot/" + TAG
if not os.path.isdir(DIR): os.makedirs(DIR)
DIR = "../checkpoint/" + TAG
if not os.path.isdir(DIR): os.makedirs(DIR)
DIR = "../eval/" + TAG + '/examples'
if not os.path.isdir(DIR): os.makedirs(DIR)
DIR = "../eval/" + TAG + '/metrics'
if not os.path.isdir(DIR): os.makedirs(DIR)

TRAIN_SET_PTH = '../dataset/your_train_set.npy'
TEST_SET_PTH  = '../dataset/your_test_set.npy'
DEV_SET_PTH   = '../dataset/your_dev_set_.npy'

model = GenGIN(in_ch=CH_INPUT, n_fea=NF_GEN, name='GIN')
opt = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.99))
