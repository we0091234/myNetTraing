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
from MyDateSets import  MyDataSets
import cv2
from torchvision.models.resnet import resnet18
from torchvision.models.resnet import resnet50
import torch.nn.functional as F
# import adabound
from lr_scheduler import LRScheduler
def distillation(y, labels, teacher_scores, temp, alpha):
    return nn.KLDivLoss()(F.log_softmax(y / temp, dim=1), F.softmax(teacher_scores / temp, dim=1)) * (
            temp * temp * 2.0 * alpha) + F.cross_entropy(y, labels) * (1. - alpha)
def cv_imread(path):
    img=cv2.imdecode(np.fromfile(path,dtype=np.uint8),-1)
    return img
# def fix_bn(m):
#     classname = m.__class__.__name__
#     if  classname.find('BatchNorm') != -1:
#         if m.num_features==32:
#            m.eval()
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
        techermodel.eval()
        for batch_idx,(img,label) in enumerate(trainloader):
                image=Variable(img.cuda())
                label=Variable(label.cuda())
                optimizer.zero_grad()
                out=model(image)
                outT=techermodel(image)
                loss=distillation(out,label,outT,temp=5.0,alpha=0.7)
                #loss=criterion(out,label)
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
	exModelName = r"/home/xiaolei/train_data/myNetTraing/model/" +str(format(accuracy,'.6f'))+"_"+"epoth_"+ str(epoch) + "_model" + ".pth.tar"
	# torch.save(model.state_dict(),exModelName)
	torch.save({'cfg': Stcfg, 'state_dict': model.state_dict()}, exModelName,_use_new_zipfile_serialization=False)

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
        parser.add_argument('--numClasses', type=int, default=3)
        parser.add_argument('--num_workers', type=int, default=4)
        parser.add_argument('--batchSize', type=int, default=512)
        parser.add_argument('--nepoch', type=int, default=120)
        parser.add_argument('--lr', type=float, default=0.025)
        parser.add_argument('--gpu', type=str, default='0')
        opt = parser.parse_args()
        print(opt)
        cfg= [32, 'M', 64, 'M', 96, 'M', 128, 'M', 192, 'M', 256]
        cfgGender = [32, 'M', 64, 'M', 76, 'M', 92, 'M', 28, 'M', 112]
        # os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
        # MEAN_NPY = r'F:\@AttributeMean\@meanFile\PedestrainGlobal.npy'
        # MEAN_NPY = r"F:\@Pedestrain_attribute\@_pedestrain2\NoStdVehicle.npy"
        # MEAN_NPY = r"F:\NostdAttribute\NS_handCar\NoStdVehicle.npy"
        MEAN_NPY ='/home/xiaolei/train_data/myNetTraing/meanFile/VehicleDriverGeneral.npy'
        mean_npy = np.load(MEAN_NPY)
        mean = mean_npy.mean(1).mean(1)
        transform_train = cvTransforms.Compose([
            cvTransforms.Resize((140, 140)),
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
        # modelPath = r"F:\Driver\DrivalBeltViolation\pruned102\0.977084_epoth_75_model.pth.tar"
        modelPath = r"/home/xiaolei/train_data/myNetTraing/model_kd/0.969970_epoth_44_model.pth.tar"
        # modelPath=r"D:\trainTemp\Driver\selfBelt\model_kd_109_3\0.913701_epoth_74_model.pth.tar"
        # modelPath=r"D:\trainTemp\Driver\selfBelt\0.906387_epoth_135_model.pth.tar"
        StmodelPath=r"/home/xiaolei/train_data/myNetTraing/model_kd/0.969970_epoth_44_model.pth.tar"
        # StmodelPath= r"D:\trainTemp\Driver\selfBelt\kdViolation4_71\0.930765_epoth_71_model.pth.tar"
        checkPoint = torch.load(modelPath)
        cfg = checkPoint["cfg"]
        # cfg = [32, 'M', 64, 'M', 96, 'M', 128, 'M', 192, 'M', 256]
        # print(cfg)
        techermodel =myNet(num_classes=opt.numClasses,cfg=cfg)
        # techermodel= nn.DataParallel(techermodel)
        techermodel.load_state_dict(checkPoint["state_dict"])
        techermodel.cuda()
        # StmodelPath = r"F:\PedestrainAttribute\gender\modelKD\0.9787045252883763_epoth_51_model.pth.tar"
        StcheckPoint = torch.load(StmodelPath)
        Stcfg = StcheckPoint["cfg"]
        print(Stcfg)
        model = myNet(num_classes=opt.numClasses, cfg=Stcfg)
        model_dict = StcheckPoint['state_dict']
        model.load_state_dict(model_dict)
        trainset = dset.ImageFolder(r'/home/xiaolei/train_data/data/datasets/trainData/DrivalCall/new/train', transform=transform_train,
                                    loader=cv_imread)
        # trainset=MyDataSets(r"C:\train_data\isDriver\label.txt",transform=transform_train)
        print(trainset[0][0])
        valset = dset.ImageFolder(r'/home/xiaolei/train_data/data/datasets/trainData/DrivalCall/new/val', transform=transform_val, loader=cv_imread)
        print(len(valset))
        trainloader=torch.utils.data.DataLoader(trainset,batch_size=opt.batchSize,shuffle=True,num_workers=opt.num_workers)
        valloader=torch.utils.data.DataLoader(valset,batch_size=opt.batchSize,shuffle=False,num_workers=opt.num_workers)	# model=myNet(num_classes=3)
        model.cuda()
        model = nn.DataParallel(model, device_ids=[0,1,2,3], output_device=0)
        name_list =['feature.0.weight','feature.0.bias','feature.1.weight','feature.1.bias','feature.4.weight','feature.4.bias','feature.5.weight','feature.5.bias']    #list??????????????????????????????
        for name, value in model.named_parameters():
        	if name in name_list:
        		value.requires_grad = False

        params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer=torch.optim.SGD(params,lr=opt.lr,momentum=0.9,weight_decay=5e-4)
        # optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9, weight_decay=5e-4)
        # optimizer = adabound.AdaBound(model.parameters(), lr=1e-3, final_lr=0.1)
        # scheduler=StepLR(optimizer,step_size=40)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 80)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(opt.nepoch))
        criterion=nn.CrossEntropyLoss()
        # criterion=CrossEntropyLabelSmooth(opt.numClasses)
        criterion.cuda()
        for epoch in range(opt.nepoch):
            train(epoch)
            val(epoch)

