import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import *
import torchvision.transforms as transforms
import cvtorchvision.cvtransforms as cvTransforms
import torchvision.datasets as dset
import torch.distributed as dist
import numpy as np 
import os
import argparse
import time
from myNet import myNet
import cv2
# from model.resnet import resnet101
# from dataset.DogCat import DogCat
def cv_imread(path):
    img=cv2.imdecode(np.fromfile(path,dtype=np.uint8),-1)
    return img
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
		# print(out.shape)
		# print(out.shape)
		out=self.fc1(out)
		# print(out.shape)
		return out

myCfg = [32, 'M', 64, 'M', 96, 'M', 128, 'M', 192, 'M', 256]

class GmyNet(nn.Module):
	def __init__(self, cfg=None, num_classes=3):
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
				conv2d = nn.Conv2d(in_channels, cfg[i], kernel_size=5, stride=1)
				if batch_norm:
					layers += [conv2d, nn.BatchNorm2d(cfg[i]), nn.ReLU(inplace=True)]
				else:
					layers += [conv2d, nn.ReLU(inplace=True)]
				in_channels = cfg[i]
			else:
				if cfg[i] == 'M':
					layers += [nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)]
				else:
					conv2d = nn.Conv2d(in_channels, cfg[i], kernel_size=3, padding=1, stride=1)
					if batch_norm:
						layers += [conv2d, nn.BatchNorm2d(cfg[i]), nn.ReLU(inplace=True)]
					else:
						layers += [conv2d, nn.ReLU(inplace=True)]
					in_channels = cfg[i]
		return nn.Sequential(*layers)

	def forward(self, x):
		x = self.feature(x)
		x = nn.AvgPool2d(2)(x)
		x = x.view(x.size(0), -1)
		y = self.classifier(x)
		return y

class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss

parser=argparse.ArgumentParser()
parser.add_argument('--num_workers',type=int,default=4)
parser.add_argument('--batchSize',type=int,default=512)
parser.add_argument('--nepoch',type=int,default=60)
parser.add_argument('--lr',type=float,default=0.025)
parser.add_argument('--gpu',type=str,default='0')
parser.add_argument('--local_rank', default=-1, type=int,
                    help='node rank for distributed training')

opt=parser.parse_args()

dist.init_process_group(backend='nccl',init_method='tcp://192.168.1.201:2000', rank=opt.local_rank, world_size=4)

torch.cuda.set_device(opt.local_rank)
word_size= torch.distributed.get_world_size()
# print(opt)
# os.environ["CUDA_VISIBLE_DEVICES"]=opt.gpu
device=torch.device("cuda")
torch.backends.cudnn.benchmark=True
# MEAN_NPY = r'C:\Train_data\@changpengzhuagnheng\@penzi\vehicle.npy'
MEAN_NPY = r'/home/xiaolei/train_data/myNetTraing/meanFile/pedestrainGlobal.npy'
# 'G:\driver_shenzhen\@new\VehicleDriverGeneral.npy'
mean_npy = np.load(MEAN_NPY)
mean = mean_npy.mean(1).mean(1)
transform_train=cvTransforms.Compose([
	cvTransforms.Resize((140,140)),
	cvTransforms.RandomCrop((128,128)),
	cvTransforms.RandomHorizontalFlip(), #镜像
	cvTransforms.ToTensorNoDiv(), #caffe中训练没有除以255所以 不除以255
	cvTransforms.NormalizeCaffe(mean)  #caffe只用减去均值
])

transform_val=cvTransforms.Compose([
	cvTransforms.Resize((128,128)),
	cvTransforms.ToTensorNoDiv(),
	cvTransforms.NormalizeCaffe(mean),
])

# def fix_bn(m):
#     classname = m.__class__.__name__
#     if  classname.find('BatchNorm') != -1:
#         if m.num_features==32:
#            m.eval()





def train(epoch,scheduler,model,trainloader,criterion,optimizer):
	
	scheduler.step()
	# print(scheduler.get_lr())
	model.train()
	# model.apply(fix_bn)

	# time_start = time.time()

	for batch_idx,(img,label) in enumerate(trainloader):
		# time_end = time.time()
		# print('totally cost', time_end - time_start)
		image=Variable(img.cuda())
		label=Variable(label.cuda())
		optimizer.zero_grad()
		out=model(image)
		loss=criterion(out,label)
		loss.backward()
		optimizer.step()
		if opt.local_rank % word_size == 0:
			if batch_idx%10==0:
				print("train Epoch:%d [%d|%d] loss:%f lr:%s" %(epoch,batch_idx,len(trainloader),loss.mean(),scheduler.get_lr()))
	# exModelName="ckp/epoth_"+str(epoch)+"_model"+".pth"
	# # torch.save(model.state_dict(),exModelName)
	# torch.save(model.state_dict(),exModelName)
def val(epoch,model,valloader):
	
	model.eval()
	total=0
	correct=0
	with torch.no_grad():
		for batch_idx,(img,label) in enumerate(valloader):
			image=Variable(img.cuda())
			label=Variable(label.cuda())
			out=model(image)
			_,predicted=torch.max(out.data,1)
			total+=image.size(0)
			correct+=predicted.data.eq(label.data).cpu().sum()
	accuracy=1.0*correct.numpy()/total
	if opt.local_rank % word_size == 0:
		print("\nValidation Epoch: %d" %epoch)
		print("Acc: %f "% ((1.0*correct.numpy())/total))
	exModelName = r"/home/xiaolei/train_data/myNetTraing/model/" +str(format(accuracy,'.6f'))+"_"+"epoth_"+ str(epoch) + "_model" + ".pth.tar"
		# torch.save(model.state_dict(),exModelName)
	torch.save({'cfg': myCfg, 'state_dict': model.module.state_dict()}, exModelName,_use_new_zipfile_serialization=False)

if __name__ == '__main__':
	trainset = dset.ImageFolder(r'/home/xiaolei/train_data/myNetTraing/datasets/gender/train1', transform=transform_train,loader=cv_imread)
	train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
	valset = dset.ImageFolder(r'/home/xiaolei/train_data/myNetTraing/datasets/gender/val', transform=transform_val,loader=cv_imread)
	val_sampler = torch.utils.data.distributed.DistributedSampler(valset)
	# print(len(valset))
	train_loader = torch.utils.data.DataLoader(trainset,
                                               batch_size=opt.batchSize,
                                               num_workers=4,
                                               pin_memory=True,
                                               sampler=train_sampler)
	# trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.batchSize, shuffle=True,
	# 										  num_workers=opt.num_workers)
	# valloader = torch.utils.data.DataLoader(valset, batch_size=opt.batchSize, shuffle=False,
	# 										num_workers=opt.num_workers)
	val_loader = torch.utils.data.DataLoader(valset,
                                             batch_size=opt.batchSize,
                                             num_workers=4,
                                             pin_memory=True,
                                             sampler=val_sampler)     
	# pretrained = torch.load("./model/result/0.880952_epoth_65_model.pth.tar")
	# pretrainedDict = pretrained['state_dict']

	myCfg = [32, 'M', 64, 'M', 96, 'M', 128, 'M', 192, 'M', 256]
	model = myNet(num_classes=2,cfg=myCfg)
   

	# model.load_state_dict(pretrainedDict)

	model.cuda(opt.local_rank)
	model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[opt.local_rank])
	# model.cuda()
	optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9, weight_decay=5e-4)
	# scheduler=StepLR(optimizer,step_size=20)
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(opt.nepoch))
	# criterion=nn.CrossEntropyLoss()
	criterion = CrossEntropyLabelSmooth(2)
	criterion.cuda(opt.local_rank)

	for epoch in range(opt.nepoch):
		train(epoch,scheduler,model,train_loader,criterion,optimizer)
		val(epoch,model,val_loader)

