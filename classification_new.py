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
import torch.nn.functional as F
from PIL import Image
def cv_imread(path):
    img=cv2.imdecode(np.fromfile(path,dtype=np.uint8),-1)
    return img
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



modelPath = r"F:\@Pedestrain_attribute\@_driver_combined\beltViolation\0.940517_epoth_52_model.pth.tar"
cfg=torch.load(modelPath)["cfg"]
# cfg = [32, 'M', 64, 'M', 96, 'M', 128, 'M', 192, 'M', 256]
model=myNet(num_classes=2,cfg=cfg)
model.load_state_dict(torch.load(modelPath)["state_dict"])
# model=myNet(num_classes=2,cfg=cfg)
model.cuda()
model.eval()
results=[]
i = 0
# foldname=r"D:\trainTemp\Upcolor\train"
# foldname=r"I:\datasets\danger_detection\results_0416"
foldname=r"D:\hz_object\bin64\release\driverSamllpic\zhuwei"
# foldname=r"M:\zhagnpengData\onePic"
# foldname=r"H:\@pedestrain_datasets\@_pedestrain1\backpack\test_171000"M:\zhagnpengData\onePic
# foldname=r"H:\CutNostd\results_taizhou"
# foldname=r"H:\CutNostd\results_zijipai_head"
toPath = r"F:\@Pedestrain_attribute\@_driver_combined\belt\52_xian"
# meanfile=r"H:\@pedestrain_datasets\sex\mean.npy"
# meanfile=r"H:\zxy_look_net\@_release\20190919\PedestrainHead.npy"
meanfile=r"G:\driver_shenzhen\@new\VehicleDriverGeneral.npy"
attr =r"0"
mean_npy = np.load(meanfile)
mean = mean_npy.mean(1).mean(1)
filelist = []
allFilePath(foldname, filelist)
num =0
if not os.path.exists(toPath):
    os.mkdir(toPath)
for imgfile in filelist:
    num+=1
    imag = cv_imread(imgfile)
    print(imgfile,type(imag))
    # if type(imag)=="NoneType":
    #     continue
    # print(imgfile)
    try:
        imag = cv2.resize(imag, (128, 128))
        imag = np.transpose(imag, (2, 0, 1))
        imag = imag.astype(np.float32)
        for i in range(3):
            imag[i, :, :] = imag[i, :, :] - mean[i]
        imag = imag.reshape(1, 3, 128, 128)
        imag = torch.from_numpy(imag)
        imag = Variable(imag.cuda())
        out =F.softmax( model(imag))
        _, predicted = torch.max(out.data, 1)
        out=out.data.cpu().numpy().tolist()
        predicted = predicted.data.cpu().numpy().tolist()
        print(num,imgfile,predicted[0])
        folderName = os.path.join(toPath, str(predicted[0]))
        prob=format(out[0][predicted[0]], '.6f')
        if not os.path.exists(folderName) :
            os.mkdir(folderName)
        if  attr and str(predicted[0]) in attr:
            imageName = imgfile.split("\\")[-1]
            imageName1=str(prob)+"_"+imageName
            toName = os.path.join(folderName, imageName1)
            shutil.copy(imgfile, toName)
        elif not attr:
            imageName = imgfile.split("\\")[-1]
            imageName1 = str(prob) + "_" + imageName
            toName = os.path.join(folderName, imageName1)
            shutil.copy(imgfile, toName)
    except:
        print("%s is error"%(imgfile))


