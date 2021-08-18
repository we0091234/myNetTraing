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



modelPath = r"C:\Train_data\@changpengzhuagnheng\@changpeng\model0811_kd\0.997191_epoth_102_model.pth.tar"
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
# foldname=r"F:\PedestrainAttribute\onePic"
# foldname=r"F:\PedestrainAttribute\株洲\taizhou"
# foldname=r"F:\PedestrainAttribute\株洲\subImg_person"
# foldname=r"H:\@pedestrain_datasets\@_pedestrain1\backpack\test_171000"M:\zhagnpengData\onePic
# foldname=r"F:\Driver\self_belt\XIANZHUFU\1"
# foldname=r"F:\Driver\self_belt\@zhuwei_xianTEST"
# foldname=r"F:\Driver\self_belt\XIANZHUFU\0"
# foldname=r"F:\Driver\self_belt\results_0416"
# foldname=r"D:\hz_object\bin64\release\driverSamllpic\zhuwei"
foldname=r"J:\@YueB\vehicle"
# foldname=r"M:\zhagnpengData\onePic"
# foldname=r"H:\@pedestrain_datasets\@_pedestrain1\backpack\test_171000"M:\zhagnpengData\onePic
toPath = r"J:\@YueB\result"
# meanfile=r"H:\@pedestrain_datasets\sex\mean.npy"
# meanfile=r"G:\driver_shenzhen\@new\VehicleDriverGeneral.npy"
# meanfile=r"F:\@Pedestrain_attribute\@_pedestrain2\NoStdVehicle.npy"
meanfile=r'C:\Train_data\@changpengzhuagnheng\@penzi\vehicle.npy'
# meanfile=r"F:\PedestrainHeadAttribute\PedestrainHead.npy"
attr =r"1"
mean_npy = np.load(meanfile)
mean = mean_npy.mean(1).mean(1)
filelist = []
allFilePath(foldname, filelist)
num =0
if not os.path.exists(toPath):
    os.mkdir(toPath)
for imgfile in filelist:
    if not imgfile.endswith(".jpg"):
        continue
    num+=1
    try:
        imag = cv_imread(imgfile)
        print(imgfile,type(imag))
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
        if not str(predicted[0]) in attr:
            if float(prob) < 0.9:
                imageName = imgfile.split("\\")[-1]
                imageName1 = str(prob) + "_" + imageName
                toName = os.path.join(folderName, imageName1)
                shutil.copy(imgfile, toName)
    except:
        print("cv error %s"%imgfile)


