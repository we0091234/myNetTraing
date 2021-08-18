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
myCfg = [32,'M',64,'M',96,'M',128,'M',192,'M',256]
class GmyNet(nn.Module):
    def __init__(self,cfg=None,num_classes=3):
        super(myNet, self).__init__()
        if cfg is None:
            cfg = myCfg
        self.feature = self.make_layers(cfg, True)
        self.classifier = nn.Linear(cfg[-1], num_classes)
    def make_layers(self, cfg, batch_norm=True):
        layers = []
        in_channels = 3
        for i in range(len(cfg)):
            if i == 0:
                conv2d =nn.Conv2d(in_channels, cfg[i], kernel_size=5,stride =1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(cfg[i]), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = cfg[i]
            else:
                if cfg[i] == 'M':
                    layers += [nn.MaxPool2d(kernel_size=3, stride=2,ceil_mode=True)]
                else:
                    conv2d = nn.Conv2d(in_channels, cfg[i], kernel_size=3, padding=1,stride =1)
                    if batch_norm:
                        layers += [conv2d, nn.BatchNorm2d(cfg[i]), nn.ReLU(inplace=True)]
                    else:
                        layers += [conv2d, nn.ReLU(inplace=True)]
                    in_channels = cfg[i]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.feature(x)
        x = nn.AvgPool2d(kernel_size=3, stride=1)(x)
        x = x.view(x.size(0), -1)
        y = self.classifier(x)
        return y
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
model=myNet()
model.load_state_dict(torch.load(r'F:\safe_belt\@driver_call\call_0905_3lei\pytorch_model1\0.9592863677950595_epoth_39_model.pth'))
model.cuda()
model.eval()
# newmodel = myNet()
# list1 = list(model.modules())
# print(list1)
# newmodel=torch.load(r'I:\BaiduNetdiskDownload\dogsAndCats\Kaggle-Dogs_vs_Cats_PyTorch-master\ckp\epoth_0.9556267154620312_179_model.pth')
# modelParamaters=[]
# for m0 in model.modules():
#     haha =[]
#     if isinstance(m0, nn.BatchNorm2d):
#         haha.append(m0.weight.data)
#         haha.append(m0.bias.data)
#         haha.append(m0.running_mean.data)
#         haha.append(m0.running_var.data)
#         modelParamaters.append(haha)
#     elif isinstance(m0, nn.Conv2d):
#         haha.append(m0.weight.data)
#         haha.append(m0.bias.data)
#         modelParamaters.append(haha)
#     elif isinstance(m0, nn.Linear):
#         haha.append(m0.weight.data)
#         haha.append(m0.bias.data)
#         modelParamaters.append(haha)
# print(len(modelParamaters))
#
# i = 0
# for m1 in newmodel.modules():
#     if isinstance(m1, nn.BatchNorm2d):
#         m1.weight.data = modelParamaters[i][0]
#         m1.bias.data = modelParamaters[i][1]
#         m1.running_mean = modelParamaters[i][2]
#         m1.running_var = modelParamaters[i][3]
#         i=i+1
#     elif isinstance(m1, nn.Conv2d):
#         m1.weight.data = modelParamaters[i][0]
#         m1.bias.data = modelParamaters[i][1]
#         i=i+1
#     elif isinstance(m1, nn.Linear):
#         m1.weight.data = modelParamaters[i][0]
#         m1.bias.data = modelParamaters[i][1]
# newmodel.cuda()
# newmodel.eval()
# # pf =open("5.txt","w")
# # for m1 in newmodel.modules():
# #     if isinstance(m1, nn.BatchNorm2d):
# #         print(type(m1.weight.data))
# #         pf.write("batch_weights" + "\n\n")
# #         a=m1.weight.data.cpu().numpy()
# #         b = a.ravel()
# #         for i in b:
# #             pf.write(str(i)+" ")
# #         pf.write("\n\n")
# #         pf.write("batch_bais" + "\n\n")
# #         a = m1.bias.data.cpu().numpy()
# #         b = a.ravel()
# #         for i in b:
# #             pf.write(str(i) + " ")
# #         pf.write("\n\n")
# #         pf.write("batch_mean" + "\n\n")
# #         a = m1.running_mean.cpu().numpy()
# #         b = a.ravel()
# #         for i in b:
# #             pf.write(str(i) + " ")
# #         pf.write("\n\n")
# #         pf.write("batch_var" + "\n\n")
# #         a = m1.running_var.cpu().numpy()
# #         b = a.ravel()
# #         for i in b:
# #             pf.write(str(i) + " ")
# #         pf.write("\n\n")
# #
# #     elif isinstance(m1, nn.Conv2d) or isinstance(m1, nn.Linear):
# #         pf.write("weights" + "\n\n")
# #         a = m1.weight.data.cpu().numpy()
# #         b = a.ravel()
# #         for i in b:
# #             pf.write(str(i) + " ")
# #         pf.write("\n\n")
# #
# #         pf.write("batch_bais" + "\n\n")
# #         a = m1.bias.data.cpu().numpy()
# #         b = a.ravel()
# #         for i in b:
# #             pf.write(str(i) + " ")
# #         pf.write("\n\n")
# # pf.close
# # pf.close
# # for m0,m1 in zip (model.modules(),newmodel.modules()):
# #     if isinstance(m0, nn.BatchNorm2d):
# #         m1.weight.data = m0.weight.data.clone()
# #         m1.bias.data = m0.bias.data.clone()
# #         m1.running_mean = m0.running_mean.clone()
# #         m1.running_var = m0.running_var.clone()
# #     elif isinstance(m0, nn.Conv2d):
# #         w1 = m0.weight.data.clone()
# #         w1 = w1.clone()
# #         m1.weight.data = w1.clone()
# #         m1.bias.data = m0.bias.data.clone()
# #     elif isinstance(m0, nn.Linear):
# #         m1.weight.data = m0.weight.data.clone()
# #         m1.bias.data = m0.bias.data.clone()
# #    if isinstance(m,nn.Conv2d):
# #        print(m.weight.data.shape)
# # print("_________________________")
# # for n in newmodel.modules():
# #     if isinstance(n, nn.Conv2d):
# #         print(n.weight.data.shape)
# # newmodel.cuda()
# # newmodel.eval()
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