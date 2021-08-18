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
import argparse
from myNet import myNet
import torch.nn.functional as F
from collections import OrderedDict
from PIL import Image
from repvgg import get_RepVGG_func_by_name, repvgg_model_convert


    
def cv_imread(path):
    img=cv2.imdecode(np.fromfile(path,dtype=np.uint8),-1)
    return img
def allFilePath(rootPath,allFIleList):
    fileList = os.listdir(rootPath)
    for temp in fileList:

# foldname=r"D:\trainTemp\Upcolor\train"
# foldname=r"F:\PedestrainAttribute\onePic"
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
# H:\@project\@luohu\data\1\headSmall\01
# H:\@project\@luohu\hat\hatSmall
# 'G:\driver_shenzhen\@new\VehicleDriverGeneral.npy'
parser=argparse.ArgumentParser()
parser.add_argument("--modelpath",type=str,default= r"L:\trainTemp\NostdVehicle\Nostd_Meiq\model\95.pth.tar")
parser.add_argument("--num_classes",type=int,default=2)
parser.add_argument("--meanfile",type=str,default=r'G:\driver_shenzhen\@new\VehicleDriverGeneral.npy')
parser.add_argument("--toPath",type=str,default=r"L:\nostd\output\3\cls")
parser.add_argument("--testPath",type=str,default=r"L:\nostd\output\3")
parser.add_argument("--attr",type=str,default="1")
parser.add_argument("--prob",type=float,default=0.9)
opt=parser.parse_args()
modelPath =opt.modelpath
meanfile=opt.meanfile
# meanfile=r"F:\PedestrainHeadAttribute\PedestrainHead.npy"
attr =opt.attr
mean_npy = np.load(meanfile)
mean = mean_npy.mean(1).mean(1)
print(mean)
# cfg=torch.load(modelPath)["cfg"]
# cfg = [32, 'M', 64, 'M', 96, 'M', 128, 'M', 192, 'M', 256]
# model=myNet(num_classes=opt.num_classes,cfg=cfg)
# model.load_state_dict(torch.load(modelPath)["state_dict"])
repvgg_build_func = get_RepVGG_func_by_name('RepVGG-B1')
model = repvgg_build_func(deploy=False)
# checkpoint = torch.load(modelPath)
# cfg =[32, 'M', 64, 'M', 92, 'M', 100, 'M', 36, 'M', 132]
# cfg =checkpoint["cfg"]
# cfg = [32, 'M', 64, 'M', 80, 'M', 84, 'M', 40, 'M', 200]
# print(cfg)
# net = myNet(num_classes=2,cfg=cfg)
# net.load_state_dict(checkpoint)
model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(modelPath).items()})
# model=myNet(num_classes=2,cfg=cfg)
model.cuda()
model.eval()
results=[]
i = 0
# foldname=r"F:\PedestrainAttribute\株洲\taizhou"
# foldname=r"F:\PedestrainAttribute\株洲\subImg_person"
# foldname=r"H:\@pedestrain_datasets\@_pedestrain1\backpack\test_171000"M:\zhagnpengData\onePic
# foldname=r"F:\Driver\self_belt\XIANZHUFU\1"
# foldname=r"F:\Driver\self_belt\@zhuwei_xianTEST"
# foldname=r"F:\Driver\self_belt\XIANZHUFU\0"
# foldname=r"F:\Driver\self_belt\results_0416"
# foldname=r"D:\hz_object\bin64\release\driverSamllpic\zhuwei"
foldname=opt.testPath
# foldname=r"M:\zhagnpengData\onePic"
# foldname=r"H:\@pedestrain_datasets\@_pedestrain1\backpack\test_171000"M:\zhagnpengData\onePic
toPath =opt.toPath
# meanfile=r"H:\@pedestrain_datasets\sex\mean.npy"
# meanfile=r"G:\driver_shenzhen\@new\VehicleDriverGeneral.npy"
# meanfile=r"F:\@Pedestrain_attribute\@_pedestrain2\NoStdVehicle.npy"
meanfile=opt.meanfile
# meanfile=r"F:\PedestrainHeadAttribute\PedestrainHead.npy"
attr =opt.attr
mean_npy = np.load(meanfile)
mean = mean_npy.mean(1).mean(1)
print(mean)
filelist = []
allFilePath(foldname, filelist)
num =0
defineprob=opt.prob
if not os.path.exists(toPath):
    os.mkdir(toPath)
for imgfile in filelist:
    if not imgfile.endswith(".jpg"):
        continue
    num+=1
    # try:
    imag = cv_imread(imgfile)
    print(imgfile,type(imag))
    imag = cv2.resize(imag, (96, 144))
    imag = np.transpose(imag, (2, 0, 1))
    imag = imag.astype(np.float32)
    for i in range(3):
        imag[i, :, :] = imag[i, :, :] - mean[i]
    imag = imag.reshape(1, 3, 144, 96)
    imag = torch.from_numpy(imag)
    imag = Variable(imag.cuda())
    out=model(imag)
    print(out.shape)
    h,w=out.shape
    print(h,w)
    with open(r"C:\Users\Administrator\Desktop\Reid\before_new_34_Convert_BN.txt", "w") as f:
        for i in range(w):
            num=out[0,i].cpu().detach().numpy()
            num = format(num, '.4f')
            temp = str(num)
            f.write(temp + '\n')
        f.close()
        #     for i in range(c):
        #         for j in range(h):
        #             for k in range(w):
        #                 num=reluOut[0,i,j,k].cpu().detach().numpy()
        #                 # if num !=0:
        #                 num=format(num,'.4f')
        #                 temp =str(num)
        #                 # if temp == '0.0':
        #                 #     temp='0'
        #                 f.write(temp+'\n')
        #         f.write("\n")
        # out =F.softmax( model(imag))
        # _, predicted = torch.max(out.data, 1)
        # out=out.data.cpu().numpy().tolist()
        # predicted = predicted.data.cpu().numpy().tolist()
        # print(num,imgfile,predicted[0])
        # folderName = os.path.join(toPath, str(predicted[0]))
        # prob=format(out[0][predicted[0]], '.6f')
        # if not os.path.exists(folderName) :
        #     os.mkdir(folderName)
        # if  attr and str(predicted[0]) in attr:
        #     # if float(prob) < 1:
        #         imageName = imgfile.split("\\")[-1]
        #         imageName1=str(prob)+"_"+imageName
        #         toName = os.path.join(folderName, imageName1)
        #         shutil.copy(imgfile, toName)
        # elif not attr:
        #     imageName = imgfile.split("\\")[-1]
        #     imageName1 = str(prob) + "_" + imageName
        #     toName = os.path.join(folderName, imageName1)
        #     shutil.copy(imgfile, toName)
        # if not str(predicted[0]) in attr:
        #     if float(prob) <defineprob:
        #         imageName = imgfile.split("\\")[-1]
        #         imageName1 = str(prob) + "_" + imageName
        #         toName = os.path.join(folderName, imageName1)
        #         shutil.copy(imgfile, toName)
    # except:
    #     print("cv error %s"%imgfile)


