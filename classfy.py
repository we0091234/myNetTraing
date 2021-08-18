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
import cv2
from myNet import myNet
from PIL import Image
def allFilePath(rootPath,allFIleList):
    fileList = os.listdir(rootPath)
    for temp in fileList:
        if os.path.isfile(os.path.join(rootPath,temp)):
            allFIleList.append(os.path.join(rootPath,temp))
        else:
            allFilePath(os.path.join(rootPath,temp),allFIleList)

class Category():
    def __init__(self):
           self.sum = 0
           self.right=0
           self.error=0
           self.rightRatio=0

model=myNet(num_classes=2)
model.load_state_dict(torch.load(r'F:\@Pedestrain_attribute\@_driver_combined\beltViolation\0.9751340809361287_epoth_112_model.pth.tar'))
model.cuda()
model.eval()
results=[]
i = 0
# foldname=r"F:\SLEEVE\subImg_person"
foldname=r"H:\二级车辆定位\1_driverSmall\2"
# foldname=r"M:\zhagnpengData\onePic"
# foldname=r"H:\@pedestrain_datasets\@_pedestrain1\backpack\test_171000"M:\zhagnpengData\onePic
toPath = r"F:\@Pedestrain_attribute\@_driver_combined\beltViolation\cls_belt_112"
# meanfile=r"H:\@pedestrain_datasets\sex\mean.npy"
meanfile=r"G:\driver_shenzhen\@new\VehicleDriverGeneral.npy"
attr =r"0"
mean_npy = np.load(meanfile)
mean = mean_npy.mean(1).mean(1)
filelist = []
allFilePath(foldname, filelist)
num =0
for imgfile in filelist:
    num+=1
    imag = cv2.imread(imgfile)
    # print(imgfile)
    imag = cv2.resize(imag, (128, 128))
    imag = np.transpose(imag, (2, 0, 1))
    imag = imag.astype(np.float32)
    for i in range(3):
        imag[i, :, :] = imag[i, :, :] - mean[i]
    imag = imag.reshape(1, 3, 128, 128)
    imag = torch.from_numpy(imag)
    imag = Variable(imag.cuda())
    out = model(imag)
    _, predicted = torch.max(out.data, 1)
    predicted = predicted.data.cpu().numpy().tolist()
    print(num,imgfile,predicted[0])
    folderName = os.path.join(toPath, str(predicted[0]))
    if not os.path.exists(folderName) :
        os.mkdir(folderName)
    if  attr and str(predicted[0]) in attr:
        imageName = imgfile.split("\\")[-1]
        toName = os.path.join(folderName, imageName)
        shutil.copy(imgfile, toName)
    elif not attr:
        imageName = imgfile.split("\\")[-1]
        toName = os.path.join(folderName, imageName)
        shutil.copy(imgfile, toName)


