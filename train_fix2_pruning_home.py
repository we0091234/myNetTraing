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
import cv2
from torchvision.models.resnet import resnet18
from torchvision.models.resnet import resnet50
# import adabound
from lr_scheduler import LRScheduler
def cv_imread(path):
    img=cv2.imdecode(np.fromfile(path,dtype=np.uint8),-1)
    return img
def cvImageLoader(path):
    # try:
    #      img = cv2.imread(path)
    #      if type(img)=="NoneType":
    #          print("error")
    # except:
    #     print("cannot open %s"%path)
    # return img
    with open(path, 'rb') as f:
        img = cv2.imread(path)
        # print(path)
        return img
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

# def fix_bn(m):
#     classname = m.__class__.__name__
#     if classname.find('BatchNorm') != -1:
#         if m.num_features == 32 or m.num_features == 64:
#             m.eval()
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
		# print("Epoch:%d [%d|%d] loss:%f lr:%s" %(epoch,batch_idx,len(trainloader),loss.mean(),scheduler.get_lr()))
		if batch_idx%200==0:
			print("Epoch:%d [%d|%d] loss:%f lr:%s" % (epoch, batch_idx, len(trainloader), loss.mean(), scheduler.get_lr()))
	# exModelName="ckp/epoth_"+str(epoch)+"_model"+".pth"
	# # torch.save(model.state_dict(),exModelName)
	# torch.save(model.state_dict(),exModelName)
# def val(epoch):
# 	print("\nValidation Epoch: %d" %epoch)
# 	model.eval()
# 	total=0
# 	correct=0
# 	with torch.no_grad():
# 		for batch_idx,(img,label) in enumerate(valloader):
# 			image=Variable(img.cuda())
# 			label=Variable(label.cuda())
# 			out=model(image)
# 			_,predicted=torch.max(out.data,1)
# 			total+=image.size(0)
# 			correct+=predicted.data.eq(label.data).cpu().sum()
# 	accuracy=1.0*correct.numpy()/total
# 	print("Acc: %f "% ((1.0*correct.numpy())/total))
# 	exModelName = r"D:/trainTemp/modelresnet/" +str(accuracy)+"_"+"epoth_"+ str(epoch) + "_model" + ".pth.tar"
# 	# torch.save(model.state_dict(),exModelName)
# 	torch.save({'cfg': cfg, 'state_dict': model.state_dict()}, exModelName)
def val(epoch):
	print("\nValidation Epoch: %d" %epoch)
	model.eval()
	total=0
	correct=0
	test_loss = 0
	with torch.no_grad():
		for batch_idx,(img,label) in enumerate(valloader):
			image=Variable(img.cuda())
			label=Variable(label.cuda())
			out=model(image)
			_,predicted=torch.max(out.data,1)
			test_loss += torch.nn.functional.cross_entropy(out, label, reduction='sum').item()
			total+=image.size(0)
			correct+=predicted.data.eq(label.data).cpu().sum()
	accuracy=1.0*correct.numpy()/total
	test_loss /= len(valloader.dataset)
	print("testAcc: %f testLoss:%f"% ((1.0*correct.numpy())/total,test_loss))
	exModelName = r"D:/trainTemp/handBag/res501/" +str(accuracy)+"_"+"epoth_"+ str(epoch) + "_model" + ".pth.tar"
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

if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--numClasses', type=int, default=2)
	parser.add_argument('--num_workers', type=int, default=8)
	parser.add_argument('--batchSize', type=int, default=32)
	parser.add_argument('--nepoch', type=int, default=120)
	parser.add_argument('--lr', type=float, default=0.01)
	parser.add_argument('--gpu', type=str, default='0')
	opt = parser.parse_args()
	print(opt)
	cfg= [32, 'M', 64, 'M', 96, 'M', 128, 'M', 192, 'M', 256]
	os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
	device = torch.device("cuda")
	torch.backends.cudnn.benchmark = True
	MEAN_NPY = r'E:\pytorch\mean.npy'
	mean_npy = np.load(MEAN_NPY)
	mean = mean_npy.mean(1).mean(1)
	transform_train = cvTransforms.Compose([
		cvTransforms.Resize((144, 144)),
		cvTransforms.RandomCrop((128, 128)),
		cvTransforms.RandomHorizontalFlip(),  # ??????
		cvTransforms.ToTensorNoDiv(),  # caffe?????????????????????255?????? ?????????255
		cvTransforms.NormalizeCaffe(mean)  # caffe??????????????????
	])

	transform_val = cvTransforms.Compose([
		cvTransforms.Resize((128, 128)),
		cvTransforms.ToTensorNoDiv(),
		cvTransforms.NormalizeCaffe(mean),
	])
	# modelPath = r"I:\BaiduNetdiskDownload\dogsAndCats\Kaggle-Dogs_vs_Cats_PyTorch-master\pruneModel\pruned_fix_coatText_new0.7.pth.tar"
	# checkPoint = torch.load(modelPath)
	# cfg = checkPoint["cfg"]
	# print(cfg)
	# model = myNet(num_classes=opt.numClasses)
	model =resnet18()
	# model_dict = checkPoint['state_dict']
	# model.load_state_dict(model_dict)
	trainset=dset.ImageFolder(r'D:\trainTemp\Driver\selfBelt\train1\train',transform=transform_train,loader=cv_imread)
	print(trainset[0][0])
	valset  =dset.ImageFolder(r'D:\trainTemp\handBag\val',transform=transform_val,loader=cv_imread)
	print(len(valset))
	trainloader=torch.utils.data.DataLoader(trainset,batch_size=opt.batchSize,shuffle=True,num_workers=opt.num_workers)
	valloader=torch.utils.data.DataLoader(valset,batch_size=opt.batchSize,shuffle=False,num_workers=opt.num_workers)	# model=myNet(num_classes=3)
	model.cuda()
	# name_list =['feature.0.weight','feature.0.bias','feature.1.weight','feature.1.bias','feature.4.weight','feature.4.bias','feature.5.weight','feature.5.bias']    #list??????????????????????????????
	# for name, value in model.named_parameters():
	# 	if name in name_list:
	# 		value.requires_grad = False
	# params = filter(lambda p: p.requires_grad, model.parameters())
	# optimizer=torch.optim.SGD(params,lr=opt.lr,momentum=0.9,weight_decay=5e-4)
	optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9, weight_decay=5e-4)
	# optimizer = adabound.AdaBound(model.parameters(), lr=1e-3, final_lr=0.1)
	scheduler=StepLR(optimizer,step_size=40)
	# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(opt.nepoch))
	criterion=nn.CrossEntropyLoss()
	# criterion=CrossEntropyLabelSmooth(opt.numClasses)
	criterion.cuda()
	for epoch in range(opt.nepoch):
		train(epoch)
		val(epoch)

