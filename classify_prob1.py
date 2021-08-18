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
import torchvision.models

class IntermediateLayerGetter(nn.ModuleDict):
    """
    Module wrapper that returns intermediate layers from a model
    It has a strong assumption that the modules have been registered
    into the model in the same order as they are used.
    This means that one should **not** reuse the same nn.Module
    twice in the forward if you want this to work.
    Additionally, it is only able to query submodules that are directly
    assigned to the model. So if `model` is passed, `model.feature1` can
    be returned, but not `model.feature1.layer2`.
    Arguments:
        model (nn.Module): model on which we will extract the features
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).
    Examples::
        >>> m = torchvision.models.resnet18(pretrained=True)
        >>> # extract layer1 and layer3, giving as names `feat1` and feat2`
        >>> new_m = torchvision.models._utils.IntermediateLayerGetter(m,
        >>>     {'layer1': 'feat1', 'layer3': 'feat2'})
        >>> out = new_m(torch.rand(1, 3, 224, 224))
        >>> print([(k, v.shape) for k, v in out.items()])
        >>>     [('feat1', torch.Size([1, 64, 56, 56])),
        >>>      ('feat2', torch.Size([1, 256, 14, 14]))]
    """

    def __init__(self, model, return_layers):
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")

        orig_return_layers = return_layers
        return_layers = {k: v for k, v in return_layers.items()}
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x):
        out = OrderedDict()
        for name, module in self.named_children():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out

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

def hook_fn_forward(module, input, output):
    print(module) # 用于区分模块
    print('input', input) # 首先打印出来
    print('output', output)
    total_feat_out.append(output) # 然后分别存入全局 list 中
    total_feat_in.append(input)


parser=argparse.ArgumentParser()
parser.add_argument("--modelpath",type=str,default= r"F:\@Pedestrain_attribute\@_driver_combined\@_release\@release0608\0.940517_epoth_52_model.pth.tar")
parser.add_argument("--num_classes",type=int,default=2)
parser.add_argument("--meanfile",type=str,default=r"G:\driver_shenzhen\@new\VehicleDriverGeneral.npy")
parser.add_argument("--toPath",type=str,default=r"F:\@Pedestrain_attribute\@_driver_combined\@_release\@release0608\ori\ori52")
parser.add_argument("--testPath",type=str,default=r"M:\xiaoyu\release-drvier-backup-106-200\result_id2\主驾")
parser.add_argument("--attr",type=str,default="0")
parser.add_argument("--prob",type=float,default=0)
opt=parser.parse_args()
modelPath =opt.modelpath
cfg=torch.load(modelPath)["cfg"]
# cfg = [32, 'M', 64, 'M', 96, 'M', 128, 'M', 192, 'M', 256]
model=myNet(num_classes=opt.num_classes,cfg=cfg)
model.load_state_dict(torch.load(modelPath)["state_dict"])

total_feat_out = []
total_feat_in = []

modules = model.named_children() #
module=[x  for name,x in modules]
for sonmodule in module[0]:
    if isinstance(sonmodule,nn.ReLU):
       sonmodule.register_forward_hook(hook_fn_forward)
module[1].register_forward_hook(hook_fn_forward)

# for name, module in modules:
#     module.register_forward_hook(hook_fn_forward)
# # model=myNet(num_classes=2,cfg=cfg)
#
mean_npy=mean_npy = np.load(r'H:\@pedestrain_datasets\sex\mean.npy')
model.cuda()
model.eval()
results=[]
imgfile=r'E:\TensorRT\tensorrtx-master\yolov3\build\1\0a4e3d7d-c6a3-48f9-aa6c-4ec6296d40f6.jpg'
imag = cv_imread(imgfile)
imag = np.transpose(imag, (2, 0, 1))
imgori=imag
# print(imgfile,type(imag))
# imag = cv2.resize(imag, (128, 128))
# imag = np.transpose(imag, (2, 0, 1))

# imag = imag.astype(np.float32)
# mean = mean_npy.mean(1).mean(1)
# for i in range(3):
#     imag[i, :, :] = imag[i, :, :] - mean[i]
# imag = imag.reshape(1, 3, 128, 128)
# #    #    #    #    #    #    #    #    #    #
# # imag=imag.reshape(-1,3,opt.inputSize,opt.inputSize)#输入单个图片的大小为3*116*116不满足输入条件还少一个batch，所以将batch设置为1
# imag = torch.from_numpy(imag)
# imag = Variable(imag.cuda())
# x=F.softmax(model(imag))
#
# # for idx in range(len(total_feat_in)):
# #     print('input: ', total_feat_in[idx])
# #     print('output: ', total_feat_out[idx])
# print(total_feat_out[4].shape)
# b,c,h,w=total_feat_out[0].shape
# reluOut=total_feat_out[0]
# b,c,h,w=reluOut.shape
# b,c,h,w=imgori.shape
# with open(r"H:\daTUtest\zxc_st\@_DriverTest\shenzhenBelt\pytorch_conv1.txt","w") as f:
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
# #
# #
# #
#     f.close()
c,h,w=imgori.shape
with open(r"E:\TensorRT\tensorrtx-master\yolov3\build\1\imag_pytorch_nomean.txt","w") as f:
    for i in range(c):
        for j in range(h):
            for k in range(w):
                num=imgori[i,j,k]
                temp =str(num)
                f.write(temp+'\n')
        f.write("\n")



    f.close()
# i = 0
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
# mean = mean_npy.mean(1).mean(1)
# filelist = []
# allFilePath(foldname, filelist)
# num =0
# defineprob=opt.prob
# if not os.path.exists(toPath):
#     os.mkdir(toPath)
# for imgfile in filelist:
#     if not imgfile.endswith(".jpg"):
#         continue
#     num+=1
#     try:
#         imag = cv_imread(imgfile)
#         print(imgfile,type(imag))
#         imag = cv2.resize(imag, (128, 128))
#         imag = np.transpose(imag, (2, 0, 1))
#         imag = imag.astype(np.float32)
#         for i in range(3):
#             imag[i, :, :] = imag[i, :, :] - mean[i]
#         imag = imag.reshape(1, 3, 128, 128)
#         imag = torch.from_numpy(imag)
#         imag = Variable(imag.cuda())
#         out =F.softmax( model(imag))
#         _, predicted = torch.max(out.data, 1)
#         out=out.data.cpu().numpy().tolist()
#         predicted = predicted.data.cpu().numpy().tolist()
#         print(num,imgfile,predicted[0])
#         folderName = os.path.join(toPath, str(predicted[0]))
#         prob=format(out[0][predicted[0]], '.6f')
#         if not os.path.exists(folderName) :
#             os.mkdir(folderName)
#         if  attr and str(predicted[0]) in attr:
#             # if float(prob) < 1:
#                 imageName = imgfile.split("\\")[-1]
#                 imageName1=str(prob)+"_"+imageName
#                 toName = os.path.join(folderName, imageName1)
#                 shutil.copy(imgfile, toName)
#         elif not attr:
#             imageName = imgfile.split("\\")[-1]
#             imageName1 = str(prob) + "_" + imageName
#             toName = os.path.join(folderName, imageName1)
#             shutil.copy(imgfile, toName)
#         if not str(predicted[0]) in attr:
#             if float(prob) <defineprob:
#                 imageName = imgfile.split("\\")[-1]
#                 imageName1 = str(prob) + "_" + imageName
#                 toName = os.path.join(folderName, imageName1)
#                 shutil.copy(imgfile, toName)
#     except:
#         print("cv error %s"%imgfile)


