import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import *
import torchvision.transforms as transforms
import cvtorchvision.cvtransforms as cvTransforms
import torchvision.datasets as dset
from utils.tripletLet import  TripletLoss
import numpy as np 
import os
import argparse
import time
from myNet import myNet
from myNetT import myNetT
from center_loss import CenterLoss

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
parser.add_argument('--num_workers',type=int,default=0)
parser.add_argument('--batchSize',type=int,default=32)
parser.add_argument('--nepoch',type=int,default=90)
parser.add_argument('--lr',type=float,default=0.025)
parser.add_argument('--gpu',type=str,default='0')
opt=parser.parse_args()
print(opt)
os.environ["CUDA_VISIBLE_DEVICES"]=opt.gpu
device=torch.device("cuda")
torch.backends.cudnn.benchmark=True
transform_train=cvTransforms.Compose([
	cvTransforms.Resize((128,128)),
	cvTransforms.RandomCrop((116,116)),
	cvTransforms.RandomHorizontalFlip(), #镜像
	cvTransforms.ToTensorNoDiv(), #caffe中训练没有除以255所以 不除以255
	cvTransforms.NormalizeCaffe((125,125,125))  #caffe只用减去均值
])

transform_val=cvTransforms.Compose([
	cvTransforms.Resize((116,116)),
	cvTransforms.ToTensorNoDiv(),
	cvTransforms.NormalizeCaffe((125,125,125)),
])

# def fix_bn(m):
#     classname = m.__class__.__name__
#     if  classname.find('BatchNorm') != -1:
#         if m.num_features==32:
#            m.eval()
# criterionS=nn.CrossEntropyLoss()
# T_criterion = TripletLoss(0.3)
# def criterionF(triplet,soft,target):
# 	return criterionS(soft,target)+T_criterion(triplet,target)



trainset=dset.ImageFolder(r'F:\safe_belt\@driver_call\call_0905_3lei\train',transform=transform_train)
print(trainset[0][0])
valset  =dset.ImageFolder(r'F:\safe_belt\@driver_call\call_0905_3lei\call_correct',transform=transform_val)
print(len(valset))
trainloader=torch.utils.data.DataLoader(trainset,batch_size=opt.batchSize,shuffle=True,num_workers=opt.num_workers)
valloader=torch.utils.data.DataLoader(valset,batch_size=opt.batchSize,shuffle=False,num_workers=opt.num_workers)
model=myNetT()
model.cuda()

# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(opt.nepoch))
# criterion=nn.CrossEntropyLoss()
# T_criterion = TripletLoss(0.3)
# criterion =criterionF
# criterion.cuda()
criterion_Cross=CrossEntropyLabelSmooth(3)
criterion_Cent=CenterLoss(num_classes=3, feat_dim=2, use_gpu=True)
optimizer=torch.optim.SGD(model.parameters(),lr=opt.lr,momentum=0.9,weight_decay=5e-4)
optimizer_centloss = torch.optim.SGD(criterion_Cent.parameters(), lr=0.5)
scheduler=StepLR(optimizer,step_size=20)

def train(epoch):
	print('\nEpoch: %d' % epoch)
	scheduler.step()
	print(scheduler.get_lr())
	model.train()
	for batch_idx,(img,label) in enumerate(trainloader):
		# time_end = time.time()
		# print('totally cost', time_end - time_start)
		image=Variable(img.cuda())
		label=Variable(label.cuda())
		optimizer.zero_grad()
		optimizer_centloss.zero_grad()
		out1,out2=model(image)
		loss1=criterion_Cross(out2,label)
		loss2=criterion_Cent(out1,label)
		loss = loss1+loss2
		loss.backward()
		optimizer.step()
		print("Epoch:%d [%d|%d] loss:%f lr:%s" %(epoch,batch_idx,len(trainloader),loss.mean(),scheduler.get_lr()))
	# exModelName="ckp/epoth_"+str(epoch)+"_model"+".pth"
	# # torch.save(model.state_dict(),exModelName)
	# torch.save(model.state_dict(),exModelName)
def val(epoch):
	print("\nValidation Epoch: %d" %epoch)
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
	print("Acc: %f "% ((1.0*correct.numpy())/total))
	exModelName = "F:\\safe_belt\\@driver_call\\call_0905_3lei\\pytorch_model6\\" +str(accuracy)+"_"+"epoth_"+ str(epoch) + "_model" + ".pth"
	# torch.save(model.state_dict(),exModelName)
	torch.save(model.state_dict(), exModelName)

for epoch in range(opt.nepoch):
	train(epoch)
	val(epoch)

