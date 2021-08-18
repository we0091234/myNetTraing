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
# from squeeze_res_Backpack import SqueezeResNet_Backpack
from torchvision.models.resnet import resnet50
from torchvision.models.resnet import resnet18
import time

def allFilePath(rootPath,allFIleList):
    fileList = os.listdir(rootPath)
    for temp in fileList:
        if os.path.isfile(os.path.join(rootPath,temp)):
            allFIleList.append(os.path.join(rootPath,temp))
        else:
            allFilePath(os.path.join(rootPath,temp),allFIleList)
def cv_imread(path):
    img=cv2.imdecode(np.fromfile(path,dtype=np.uint8),-1)
    return img
class Category():
    def __init__(self):
           self.sum = 0
           self.right=0
           self.error=0
           self.rightRatio=0

parser=argparse.ArgumentParser()
parser.add_argument('--cfg',type=bool,default=True)
parser.add_argument('--numclass',type=int,default=14)
parser.add_argument('--inputSize',type=int,default=128)
parser.add_argument('--isParallel',type=bool,default=False)
parser.add_argument('--testfile',type=str,default='')
parser.add_argument('--meanfile',type=str,default='')
parser.add_argument('--modelfile',type=str,default='')
parser.add_argument('--savefile',type=str,default='')
parser.add_argument('--gpu',type=str,default='0')
opt=parser.parse_args()
# opt.meanfile =r'H:\@pedestrain_datasets\sex\mean.npy'
# opt.meanfile=r"G:\driver_shenzhen\@new\VehicleDriverGeneral.npy"
# opt.meanfile=r"H:\zxy_look_net\@_release\20190919\PedestrainHead.npy"
opt.meanfile=r'C:\Train_data\@changpengzhuagnheng\@penzi\vehicle.npy'
# opt.testfile = r"H:\@pedestrain_datasets\@_pedestrain1\baby\valreal"
# opt.testfile = r"M:\yuemingsmall2\0"
# opt.testfile = r"M:\GetSmallPIcNostd1\passenger"
# opt.testfile=r"F:\@Pedestrain_attribute\@_pedestrain2\NS-type\model0714\69_jian2\done"
# opt.testfile=r"M:\GetSmallPIcNostd1\No_stdVechileType"
opt.testfile=r"J:\@YueB\val_allreal"
# opt.testfile=r"M:\yuemingsmall2\10\1"
# opt.testfile=r"M:\GetSmallPIc2\11"
# opt.testfile=r"M:\GetSmallPIcNostd\11"
# opt.testfile=r"D:\@linux_share\val_ori"
opt.savefile =r"C:\Train_data\@changpengzhuagnheng\@specialCar"
opt.modelfile = r"C:\Train_data\@changpengzhuagnheng\@specialCar\0.943882_epoth_558_model.pth.tar"
# opt.isParallel = True
# M:\zhagnpengData\onePic
mean_npy = np.load(opt.meanfile)
mean = mean_npy.mean(1).mean(1)
if not os.path.exists(opt.savefile):
    os.mkdir(opt.savefile)
filelist = []
modellist=[]
allFilePath(opt.testfile, filelist)
sumTime=0
cfg=[]
if not os.path.isfile(opt.modelfile):#not opt.modelfile.endswith("pth.tar"):
    allFilePath(opt.modelfile,modellist)
    for modeli in modellist:
        if not modeli.endswith("pth.tar"):
            continue
        allCat=[]
        for i in range(opt.numclass):
            OneCat = Category()
            allCat.append(OneCat)
        checkPoint = torch.load(modeli)
        cfg = checkPoint['cfg']
        # # state_dict=checkPoint['state_dict']
        # model = myNet(num_classes=opt.numclass,cfg=cfg)
        if opt.cfg == True:
            cfg = checkPoint["cfg"]
            # cfg=[32, 'M', 64, 'M', 80, 'M', 84, 'M', 40, 'M', 200]
            model = myNet(num_classes=opt.numclass, cfg=cfg)
            # model =resnet50()
        else:
            model = resnet18()
        if opt.isParallel:
            model = nn.DataParallel(model)
        model.load_state_dict(checkPoint['state_dict'])
        model.cuda()
        model.eval()

        results=[]
        i = 0
        picNum = 0
        # foldname=r"F:\safe_belt\@driver_call\call_0905_3lei\call_correct"
        toPath = opt.savefile
        print(len(filelist))
        for imgfile in filelist:
            picNum+=1
            # 图片预处理和训练师保持一致
            imag = cv_imread(imgfile)
            print(imgfile)
            imag=cv2.resize(imag,(opt.inputSize,opt.inputSize))
            imag=np.transpose(imag,(2,0,1))
            imag = imag.astype(np.float32)
            for i in range(3):
                imag[i, :, :] = imag[i, :, :] - mean[i]
            imag = imag.reshape(1, 3, opt.inputSize, opt.inputSize)
            #    #    #    #    #    #    #    #    #    #
            # imag=imag.reshape(-1,3,opt.inputSize,opt.inputSize)#输入单个图片的大小为3*116*116不满足输入条件还少一个batch，所以将batch设置为1
            imag = torch.from_numpy(imag)
            imag = Variable(imag.cuda())
            out = model(imag)
            _, predicted = torch.max(out.data, 1)
            predicted = predicted.data.cpu().numpy().tolist()

            folderName = os.path.join(toPath, str(predicted[0]))
            # if not os.path.exists(folderName):
            #     os.mkdir(folderName)
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
elif opt.modelfile.endswith("pth.tar"):
    allCat = []
    for i in range(opt.numclass):
        OneCat = Category()
        OneCat.right=0
        OneCat.error=0
        OneCat.rightRatio=0
        allCat.append(OneCat)


    check_point=torch.load(opt.modelfile)
    if opt.cfg==True:
        cfg=check_point["cfg"]
        model = myNet(num_classes=opt.numclass,cfg=cfg)
        if opt.isParallel:
            model = nn.DataParallel(model)
    # model =resnet50()
    else:
        model=resnet18()
    model.load_state_dict(check_point['state_dict'])
    model.cuda()
    model.eval()

    results = []
    i = 0
    picNum = 0
    # foldname=r"F:\safe_belt\@driver_call\call_0905_3lei\call_correct"
    toPath = r"H:\@pedestrain_attribute2020_1\pytorchModel"
    print(len(filelist))
    for imgfile in filelist:
        picNum += 1
        # 图片预处理和训练师保持一致
        imag = cv_imread(imgfile)
        print(imgfile)
        imag = cv2.resize(imag, (opt.inputSize, opt.inputSize))
        imag = np.transpose(imag, (2, 0, 1))
        imag = imag.astype(np.float32)
        for i in range(3):
            imag[i, :, :] = imag[i, :, :] - mean[i]
        imag = imag.reshape(1, 3, opt.inputSize, opt.inputSize)
        #    #    #    #    #    #    #    #    #    #
        # imag = imag.reshape(-1, 3, opt.inputSize, opt.inputSize)  # 输入单个图片的大小为3*116*116不满足输入条件还少一个batch，所以将batch设置为1
        imag = torch.from_numpy(imag)
        imag = Variable(imag.cuda())
        start=time.time()
        out = model(imag)
        end =time.time()
        interval=end-start
        sumTime+=interval
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
    opicTime=sumTime/len(filelist)
    print("耗时%f秒"%opicTime)

elif opt.modelfile.endswith("pth"):
    allCat = []
    for i in range(opt.numclass):
        OneCat = Category()
        OneCat.right = 0
        OneCat.error = 0
        OneCat.rightRatio = 0
        allCat.append(OneCat)

    check_point = torch.load(opt.modelfile)
    myCfg = [32, 'M', 64, 'M', 96, 'M', 128, 'M', 192, 'M', 256]
    if opt.cfg == True:
        cfg = check_point["cfg"]
        model = myNet(num_classes=opt.numclass, cfg=cfg)
    # model =resnet50()
    else:
        model = myNet(num_classes=opt.numclass)
    if opt.isParallel:
        model = nn.DataParallel(model)
    model.load_state_dict(check_point)
    model.cuda()
    model.eval()

    results = []
    i = 0
    picNum = 0
    # foldname=r"F:\safe_belt\@driver_call\call_0905_3lei\call_correct"
    toPath = r"E:\pytorch\save"
    print(len(filelist))
    for imgfile in filelist:
        picNum += 1
        # 图片预处理和训练师保持一致
        imag = cv2.imread(imgfile)
        print(imgfile)
        imag = cv2.resize(imag, (opt.inputSize, opt.inputSize))
        imag = np.transpose(imag, (2, 0, 1))
        imag = imag.astype(np.float32)
        for i in range(3):
            imag[i, :, :] = imag[i, :, :] - mean[i]
        imag = imag.reshape(1, 3, opt.inputSize, opt.inputSize)
        #    #    #    #    #    #    #    #    #    #
        # imag = imag.reshape(-1, 3, opt.inputSize, opt.inputSize)  # 输入单个图片的大小为3*116*116不满足输入条件还少一个batch，所以将batch设置为1
        imag = torch.from_numpy(imag)
        imag = Variable(imag.cuda())
        start = time.time()
        out = model(imag)
        end = time.time()
        interval = end - start
        sumTime += interval
        _, predicted = torch.max(out.data, 1)
        predicted = predicted.data.cpu().numpy().tolist()

        folderName = os.path.join(opt.savefile, str(predicted[0]))
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
        if allCat[int(i)].sum == 0:
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
    opicTime = sumTime / len(filelist)
    print("耗时%f秒" % opicTime)


