import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import os
from torch.autograd import Variable
import argparse
import numpy as np
from torch.optim.lr_scheduler import *
import csv
import torchvision.datasets as dset
import shutil
from PIL import Image
def allFilePath(rootPath,allFIleList):
    fileList = os.listdir(rootPath)
    for temp in fileList:
        if os.path.isfile(os.path.join(rootPath,temp)):
            allFIleList.append(os.path.join(rootPath,temp))
        else:
            allFilePath(os.path.join(rootPath,temp),allFIleList)
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 5, stride=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm2d(32),
            torch.nn.MaxPool2d(kernel_size=3,stride=2,ceil_mode=True))
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, 3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm2d(64),
            torch.nn.MaxPool2d(kernel_size=3, stride=2,ceil_mode=True)
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 96, 3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm2d(96),
            torch.nn.MaxPool2d(kernel_size=3, stride=2,ceil_mode=True)
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(96, 128, 3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm2d(128),
            torch.nn.MaxPool2d(kernel_size=3, stride=2,ceil_mode=True)
        )
        self.conv5 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 192, 3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm2d(192),
            torch.nn.MaxPool2d(kernel_size=3, stride=2,ceil_mode=True)
        )
        self.conv6 = torch.nn.Sequential(
            torch.nn.Conv2d(192, 256, 3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm2d(256),
            torch.nn.AvgPool2d(kernel_size=3, stride=1)
        )
        self.fc1=torch.nn.Linear(256,3)

    def forward(self, x):
        conv1_out = self.conv1(x)
        # print(conv1_out.shape)
        conv2_out = self.conv2(conv1_out)
        # print(conv2_out.shape)
        conv3_out = self.conv3(conv2_out)
        # print(conv3_out.shape)
        conv4_out = self.conv4(conv3_out)
        # print(conv4_out.shape)
        conv5_out = self.conv5(conv4_out)
        # print(conv5_out.shape)
        out=self.conv6(conv5_out)
        # print("out={}",out.shape)
        out = out.view(out.shape[0],-1)
        # if isFc:
        #     (b, in_f) = out.shape  # 查看卷积层输出的tensor平铺后的形状
        #     self.fc1 = nn.Linear(in_f, 3)  # 全链接层输出三类
        # print(out.shape)
        # print(out.shape)
        out=self.fc1(out)
        # print(out.shape)
        return out

model=Net()
model.load_state_dict(torch.load(r'I:\BaiduNetdiskDownload\dogsAndCats\Kaggle-Dogs_vs_Cats_PyTorch-master\ckp\epoth_0.9556267154620312_179_model.pth'))
model.cuda()
model.eval()
pf =open("1.txt","w")
for m in model.modules():
    if isinstance(m, nn.Conv2d):
        print(m.weight.data)
