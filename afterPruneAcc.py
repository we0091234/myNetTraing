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
from myNet import myNet
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
allCat=[]
for i in range(3):
    OneCat = Category()
    allCat.append(OneCat)
modelPath = r"I:\BaiduNetdiskDownload\dogsAndCats\Kaggle-Dogs_vs_Cats_PyTorch-master\model\pruned.pth.tar"
checkPoint = torch.load(modelPath)
cfg = checkPoint['cfg']
model=myNet(cfg)
model.load_state_dict(checkPoint['state_dict'])
# model.load_state_dict(torch.load(r'D:\@linux_share\0.9524245196706312_epoth_21_model.pth'))
model.cuda()
model.eval()
results=[]
i = 0
foldname=r"F:\safe_belt\@driver_call\call_0905_3lei\call_correct"
toPath = r"F:\safe_belt\@driver_call\call_0905_3lei\to_path\topath"
filelist = []
allFilePath(foldname, filelist)
print(len(filelist))
for imgfile in filelist:
    # 图片预处理和训练师保持一致
    imag = Image.open(imgfile)
    imag = transforms.Compose([
    transforms.Resize((116, 116)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),])(imag)
    #    #    #    #    #    #    #    #    #    #
    imag=imag.reshape(-1,3,116,116)#输入单个图片的大小为3*116*116不满足输入条件还少一个batch，所以将batch设置为1

    imag = Variable(imag.cuda())
    out = model(imag)
    _, predicted = torch.max(out.data, 1)
    predicted = predicted.data.cpu().numpy().tolist()
    print(imgfile,predicted[0])
    folderName = os.path.join(toPath, str(predicted[0]))
    if not os.path.exists(folderName):
        os.mkdir(folderName)
    imageName = imgfile.split("\\")[-1]
    pos1=imageName.rfind(".")
    pos2=imageName.rfind("-")
    imageLabel = imageName[pos2+1:pos1]
    print(imageLabel)
    allCat[int(imageLabel)].sum=allCat[int(imageLabel)].sum+1
    if int(predicted[0])==int(imageLabel):
        allCat[int(imageLabel)].right = allCat[int(imageLabel)].right + 1
    else:
        allCat[int(imageLabel)].error = allCat[int(imageLabel)].error + 1
    toName = os.path.join(folderName, imageName)
    # shutil.copy(imgfile, toName)
allSum=0
allRightSum=0
for i in range(3):
    print("sum=%d,right=%d,error=%d,ratio=%.4f"%(allCat[i].sum,allCat[i].right,allCat[i].error,1.0* allCat[int(i)].right/allCat[int(i)].sum))
    allSum=allSum+allCat[i].sum
    allRightSum=allRightSum+allCat[i].right
print("allRatio=%.4f",1.0*allRightSum/allSum)