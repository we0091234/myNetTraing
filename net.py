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
import argparse
import cv2
import cvtorchvision.cvtransforms as cvTransforms


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

parser=argparse.ArgumentParser()
parser.add_argument('--numclass',type=int,default=11)
parser.add_argument('--isParallel',type=bool,default=False)
parser.add_argument('--testfile',type=str,default='')
parser.add_argument('--meanfile',type=str,default='')
parser.add_argument('--modelfile',type=str,default='')
parser.add_argument('--savefile',type=str,default='')
parser.add_argument('--gpu',type=str,default='0')
opt=parser.parse_args()
opt.meanfile =r"H:\@pedestrain_datasets\sex\mean.npy"
opt.testfile = r"M:\yuemingSamll\1"
# opt.testfile = r"M:\GetSmallPIc\1"
# opt.modelfile=r"D:\@linux_share\new_cpk\1_(1).pth"
opt.savefile =r"M:\yuemingSamll\yuemingResult"
opt.modelfile = r"D:\@linux_share\epoth_0.9130895601483837_epoth_447_model.pth"
# opt.isParallel = TrueM:\zhagnpengData\onePic
mean_npy = np.load(opt.meanfile)
mean = mean_npy.mean(1).mean(1)

filelist = []
modellist=[]
allFilePath(opt.testfile, filelist)
if not opt.modelfile.endswith("pth"):
    allFilePath(opt.modelfile,modellist)
    for modeli in modellist:
        if not modeli.endswith("pth"):
            continue
        allCat=[]
        for i in range(opt.numclass):
            OneCat = Category()
            allCat.append(OneCat)
        model = myNet(num_classes=opt.numclass)
        if opt.isParallel:
            model = nn.DataParallel(model)
        model.load_state_dict(torch.load(modeli))
        model.cuda()
        model.eval()

        results=[]
        i = 0
        picNum = 0
        # foldname=r"F:\safe_belt\@driver_call\call_0905_3lei\call_correct"
        toPath = r"F:\safe_belt\@driver_call\call_0905_3lei\to_path\topath"
        print(len(filelist))
        for imgfile in filelist:
            picNum+=1
            # 图片预处理和训练师保持一致
            imag = cv2.imread(imgfile)
            print(imgfile)
            imag=cv2.resize(imag,(116,116))
            imag=np.transpose(imag,(2,0,1))
            imag = imag.astype(np.float32)
            for i in range(3):
                imag[i, :, :] = imag[i, :, :] - mean[i]
            imag = imag.reshape(1, 3, 116, 116)
            #    #    #    #    #    #    #    #    #    #
            imag=imag.reshape(-1,3,116,116)#输入单个图片的大小为3*116*116不满足输入条件还少一个batch，所以将batch设置为1
            imag = torch.from_numpy(imag)
            imag = Variable(imag.cuda())
            out = model(imag)
            _, predicted = torch.max(out.data, 1)
            predicted = predicted.data.cpu().numpy().tolist()

            folderName = os.path.join(toPath, str(predicted[0]))
            if not os.path.exists(folderName):
                os.mkdir(folderName)
            imageName = imgfile.split("\\")[-1]
            pos1=imageName.rfind(".")
            pos2=imageName.rfind("-")
            imageLabel = imageName[pos2+1:pos1]
            print(picNum,imgfile,imageLabel,predicted[0])
            # print(imageLabel)
            allCat[int(imageLabel)].sum=allCat[int(imageLabel)].sum+1
            if int(predicted[0])==int(imageLabel):
                allCat[int(imageLabel)].right = allCat[int(imageLabel)].right + 1
            else:
                allCat[int(imageLabel)].error = allCat[int(imageLabel)].error + 1
            toName = os.path.join(folderName, imageName)
            # shutil.copy(imgfile, toName)
        allSum=0
        allRightSum=0
        allErrorSum=0
        strlist = []
        for i in range(opt.numclass):
            if allCat[i].sum==0:
                continue
            allCat[i].rightRatio=1.0* allCat[int(i)].right/allCat[int(i)].sum
            allSum=allSum+allCat[i].sum
            allRightSum=allRightSum+allCat[i].right
            allErrorSum=allErrorSum+allCat[i].error
            strlist.append(
            "{:<10d} sum:{:<10d} right:{:<10d} error:{:<10d} ratio:{:<10f}".format(i,allCat[i].sum, allCat[i].right, allCat[i].error,
                                                                           allCat[i].rightRatio))
        allSumRatio=1.0 * allRightSum / allSum
        strlist.append("{:<10s} sum:{:<10d} right:{:<10d} error:{:<10d} ratio:{:<10f}".format("all",allSum, allRightSum, allErrorSum,
                                                                             allSumRatio))
        txtName="{:.4f}_".format(allSumRatio)
        for i in range(len(strlist)-1):
            # if(i!=2):
            txtName+="{:.4f}_".format(allCat[i].rightRatio)
            print(i)
        model_name =modeli[modeli.rfind("\\")+1:modeli.rfind(".")]
        testFolderName = opt.testfile.split("\\")[-1]
        txtName+=model_name+"_"+testFolderName+".txt"
        print(txtName)
        txtPath=os.path.join(opt.savefile,txtName)
        with open(txtPath,"w") as f:
            for i in range(len(strlist)):
                f.write(strlist[i]+"\n")
            f.close()
else:
    allCat = []
    for i in range(opt.numclass):
        OneCat = Category()
        OneCat.right=0
        OneCat.error=0
        OneCat.rightRatio=0
        allCat.append(OneCat)
    model = myNet(num_classes=opt.numclass)
    if opt.isParallel:
        model = nn.DataParallel(model)
    model.load_state_dict(torch.load(opt.modelfile))
    model.cuda()
    model.eval()

    results = []
    i = 0
    picNum = 0
    # foldname=r"F:\safe_belt\@driver_call\call_0905_3lei\call_correct"
    toPath = r"F:\safe_belt\@driver_call\call_0905_3lei\to_path\topath"
    print(len(filelist))
    for imgfile in filelist:
        picNum += 1
        # 图片预处理和训练师保持一致
        imag = cv2.imread(imgfile)
        print(imgfile)
        imag = cv2.resize(imag, (116, 116))
        imag = np.transpose(imag, (2, 0, 1))
        imag = imag.astype(np.float32)
        for i in range(3):
            imag[i, :, :] = imag[i, :, :] - mean[i]
        imag = imag.reshape(1, 3, 116, 116)
        #    #    #    #    #    #    #    #    #    #
        imag = imag.reshape(-1, 3, 116, 116)  # 输入单个图片的大小为3*116*116不满足输入条件还少一个batch，所以将batch设置为1
        imag = torch.from_numpy(imag)
        imag = Variable(imag.cuda())
        out = model(imag)
        _, predicted = torch.max(out.data, 1)
        predicted = predicted.data.cpu().numpy().tolist()

        folderName = os.path.join(toPath, str(predicted[0]))
        if not os.path.exists(folderName):
            os.mkdir(folderName)
        imageName = imgfile.split("\\")[-1]
        pos1 = imageName.rfind(".")
        pos2 = imageName.rfind("-")
        imageLabel = imageName[pos2 + 1:pos1]
        print(picNum, imgfile, imageLabel, predicted[0])
        # print(imageLabel)
        allCat[int(imageLabel)].sum = allCat[int(imageLabel)].sum + 1
        if int(predicted[0]) == int(imageLabel):
            allCat[int(imageLabel)].right = allCat[int(imageLabel)].right + 1
        else:
            allCat[int(imageLabel)].error = allCat[int(imageLabel)].error + 1
        toName = os.path.join(folderName, imageName)
        # shutil.copy(imgfile, toName)
    allSum = 0
    allRightSum = 0
    allErrorSum = 0
    strlist = []
    for i in range(opt.numclass):
        if allCat[int(i)].sum==0:
            continue
        allCat[i].rightRatio = 1.0 * allCat[int(i)].right / allCat[int(i)].sum
        allSum = allSum + allCat[i].sum
        allRightSum = allRightSum + allCat[i].right
        allErrorSum = allErrorSum + allCat[i].error
        strlist.append(
            "{:<10d} sum:{:<10d} right:{:<10d} error:{:<10d} ratio:{:<10f}".format(i, allCat[i].sum,
                                                                                   allCat[i].right,
                                                                                   allCat[i].error,
                                                                                   allCat[i].rightRatio))
    allSumRatio = 1.0 * allRightSum / allSum
    strlist.append(
        "{:<10s} sum:{:<10d} right:{:<10d} error:{:<10d} ratio:{:<10f}".format("all", allSum, allRightSum,
                                                                               allErrorSum,
                                                                               allSumRatio))
    txtName = "{:.4f}_".format(allSumRatio)
    for i in range(len(strlist) - 1):
        # if(i!=2):
        txtName += "{:.4f}_".format(allCat[i].rightRatio)
        print(i)
    model_name = opt.modelfile[opt.modelfile.rfind("\\") + 1:opt.modelfile.rfind(".")]
    testFolderName = opt.testfile.split("\\")[-1]
    txtName += model_name + "_" + testFolderName + ".txt"
    print(txtName)
    txtPath = os.path.join(opt.savefile, txtName)
    with open(txtPath, "w") as f:
        for i in range(len(strlist)):
            f.write(strlist[i] + "\n")
        f.close()


