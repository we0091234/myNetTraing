import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import *
import torchvision.transforms as transforms
import cvtorchvision.cvtransforms as cvTransforms
import torchvision.datasets as dset
import numpy as np 
import os
import argparse
import time
from myNet import myNet
import torch.nn.functional as F
# from myNetFace import myNetFace
import adabound
from lr_scheduler import LRScheduler
from torch.nn import Parameter
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

class ArcMarginModel(nn.Module):
    def __init__(self):
        super(ArcMarginModel, self).__init__()

        self.weight = Parameter(torch.FloatTensor(2, 256))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = False
        self.m = 0.5
        self.s = 64

        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

    def forward(self, input, label):
        x = F.normalize(input)
        W = F.normalize(self.weight)
        cosine = F.linear(x, W)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m  # cos(theta + m)
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        #one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output

def fix_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        if m.num_features == 32 or m.num_features == 64:
            m.eval()
def train(epoch):
	print('\nEpoch: %d' % epoch)


    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # lr_scheduler = LRScheduler(base_lr=3e-2, step=[30, 60, 120],
		# 					   factor=0.1, warmup_epoch=10,
		# 					   warmup_begin_lr=3e-4)
    #
    # lr = lr_scheduler.update(epoch)
    # for name, value in model.named_parameters():
		# if name in name_list:
		# 	value.requires_grad = False
    # params = filter(lambda p: p.requires_grad, model.parameters())
    # print(params)
    # optimizer = optim.SGD(params, lr=lr, weight_decay=5e-4, momentum=0.9, nesterov=True)
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
	scheduler.step()
	print(scheduler.get_lr())
	model.train()
	model.apply(fix_bn)

	# time_start = time.time()

	for batch_idx,(img,label) in enumerate(trainloader):
		# time_end = time.time()
		# print('totally cost', time_end - time_start)
		image=Variable(img.cuda())
		label=Variable(label.cuda())
		optimizer.zero_grad()
		# out1,out2=model(image)
		# loss1=criterion(out1,label)
		# output2=metric_fc(out2,label)
		# loss2 = criterion(output2,label)
		# loss = 0.5 * loss1 + 0.5 * loss2
		out = model(image)
		loss = criterion(out, label)
		loss.backward()
		optimizer.step()
		# print("Epoch:%d [%d|%d] loss:%f lr:%s" %(epoch,batch_idx,len(trainloader),loss.mean(),scheduler.get_lr()))
		if batch_idx %200==0:
			print("Epoch:%d [%d|%d] loss:%f lr:%s" % (epoch, batch_idx, len(trainloader), loss.mean(), scheduler.get_lr()))
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
			out1 = model(image)
			# out1,out2=model(image) face
			_,predicted=torch.max(out1.data,1)
			total+=image.size(0)
			correct+=predicted.data.eq(label.data).cpu().sum()
	accuracy=1.0*correct.numpy()/total
	print("Acc: %f "% ((1.0*correct.numpy())/total))
	exModelName = savePath +str(accuracy)+"_"+"epoth_"+ str(epoch) + "_model" + ".pth.tar"
	# torch.save(model.state_dict(),exModelName)
	torch.save({'cfg': cfg, 'state_dict': model.state_dict()}, exModelName)


# with open("new1.txt","w") as pf:
# 	for k,v in pretrained_dict.items():
# 		pf.write("\n"+k + "\n")
# 		a = v.cpu().numpy()
# 		b = a.ravel()
# 		for i in b:
# 			pf.write(str(i) + " ")
# 			# pf.write("\n\n")


#/////////////////////////////////////////////////////////////////////////////////////////////////


# def fix_bn(m):
#     classname = m.__class__.__name__
#     if  classname.find('BatchNorm') != -1:
#         if m.num_features==32:
#            m.eval()
# MEAN_NPY = r"F:\@Pedestrain_attribute\@_pedestrain2\NoStdVehicle.npy"
# MEAN_NPY = r"G:\driver_shenzhen\@new\VehicleDriverGeneral.npy"
# F:\@AttributeMean\@meanFile\PedestrainHead.npy
MEAN_NPY = r"F:\@AttributeMean\@meanFile\PedestrainGlobal.npy"
savePath = r"C:/train_data/6_year/model0107/"
if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--numClasses', type=int, default=4)
	parser.add_argument('--num_workers', type=int, default=4)
	parser.add_argument('--batchSize', type=int, default=128)
	parser.add_argument('--nepoch', type=int, default=120)
	parser.add_argument('--lr', type=float, default=0.025)
	parser.add_argument('--gpu', type=str, default='0')
	opt = parser.parse_args()
	print(opt)
	os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
	device = torch.device("cuda")

	torch.backends.cudnn.benchmark = True

	mean_npy = np.load(MEAN_NPY)
	mean = mean_npy.mean(1).mean(1)
	transform_train = cvTransforms.Compose([
		cvTransforms.Resize((140, 140)),
		cvTransforms.RandomCrop((128, 128)),
		cvTransforms.RandomHorizontalFlip(),  # 镜像
		cvTransforms.ToTensorNoDiv(),  # caffe中训练没有除以255所以 不除以255
		cvTransforms.NormalizeCaffe(mean)  # caffe只用减去均值
	])

	transform_val = cvTransforms.Compose([
		cvTransforms.Resize((128, 128)),
		cvTransforms.ToTensorNoDiv(),
		cvTransforms.NormalizeCaffe(mean),
	])
	modelPath = r"C:\train_data\6_year\0.682692_epoth_75_model.pth.tar"

	checkPoint = torch.load(modelPath)
	cfg = checkPoint["cfg"]
	print(cfg)
	model = myNet(num_classes=opt.numClasses, cfg=cfg)
	# model =myNetFace(num_classes=opt.numClasses, cfg=cfg)
	metric_fc = ArcMarginModel()
	metric_fc.cuda()
	model_dict = checkPoint['state_dict']
	model.load_state_dict(model_dict)
	trainset=dset.ImageFolder(r'C:\train_data\6_year\train_3lei',transform=transform_train)
	print(trainset[0][0])
	valset  =dset.ImageFolder(r'H:\daTUtest\zxc_st\@luohuTest\luohuKids\cls4',transform=transform_val)
	print(len(valset))
	trainloader=torch.utils.data.DataLoader(trainset,batch_size=opt.batchSize,shuffle=True,num_workers=opt.num_workers)
	valloader=torch.utils.data.DataLoader(valset,batch_size=opt.batchSize,shuffle=False,num_workers=opt.num_workers)	# model=myNet(num_classes=3)
	model.cuda()
	name_list =['feature.0.weight','feature.0.bias','feature.1.weight','feature.1.bias','feature.4.weight','feature.4.bias','feature.5.weight','feature.5.bias']    #list中为需要冻结的网络层
	for name, value in model.named_parameters():
		if name in name_list:
			value.requires_grad = False
	params = filter(lambda p: p.requires_grad, model.parameters())
	optimizer=torch.optim.SGD(params,lr=opt.lr,momentum=0.9,weight_decay=5e-4)
	# optimizer = adabound.AdaBound(model.parameters(), lr=1e-3, final_lr=0.1)
	# scheduler=StepLR(optimizer,step_size=40)
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(opt.nepoch))
	# criterion=nn.CrossEntropyLoss()
	criterion=CrossEntropyLabelSmooth(opt.numClasses)
	criterion.cuda()
	for epoch in range(opt.nepoch):
		train(epoch)
		val(epoch)

