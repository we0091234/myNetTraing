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
from myNet import myNet
import torchvision.datasets as dset
from model.resnet import resnet101
from dataset.DogCat import DogCat
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

parser = argparse.ArgumentParser()
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--batchSize', type=int, default=64)
parser.add_argument('--nepoch', type=int, default=1, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--gpu', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
opt = parser.parse_args()
print(opt)
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu

transform_test = transforms.Compose([
    transforms.Resize((116,116)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

testset=dset.ImageFolder(r'F:\safe_belt\@driver_call\call_0905_3lei\call_correct',transform=transform_test)
testloader=torch.utils.data.DataLoader(testset,batch_size=opt.batchSize,shuffle=False,num_workers=opt.num_workers)
model=Net()
model.load_state_dict(torch.load('I:\BaiduNetdiskDownload\dogsAndCats\Kaggle-Dogs_vs_Cats_PyTorch-master\ckp\epoth_0.9556267154620312_179_model.pth'))
model.cuda()
model.eval()
total = 0
correct = 0
with torch.no_grad():
    for image,label in testloader:
        image=Variable(image.cuda())
        label=Variable(label.cuda())
        out=model(image)
        # label=label.numpy().tolist()
        _,predicted=torch.max(out.data,1)
        # predicted=predicted.data.cpu().numpy().tolist()
        total += image.size(0)
        correct += predicted.data.eq(label.data).cpu().sum()
    print("Acc: %f " % ((1.0 * correct.numpy()) / total))


