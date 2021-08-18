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

parser=argparse.ArgumentParser()
parser.add_argument('--num_workers',type=int,default=0)
parser.add_argument('--batchSize',type=int,default=32)
parser.add_argument('--nepoch',type=int,default=90)
parser.add_argument('--lr',type=float,default=0.025)
parser.add_argument('--gpu',type=str,default='0')
opt=parser.parse_args()
device=torch.device("cuda")
torch.backends.cudnn.benchmark=True

def allFilePath(rootPath,allFIleList):
    fileList = os.listdir(rootPath)
    for temp in fileList:
        if os.path.isfile(os.path.join(rootPath,temp)):
            allFIleList.append(os.path.join(rootPath,temp))
        else:
            allFilePath(os.path.join(rootPath,temp),allFIleList)


check_point =torch.load(r"I:\BaiduNetdiskDownload\dogsAndCats\Kaggle-Dogs_vs_Cats_PyTorch-master\model\pruned.pth.tar")
cfg = check_point["cfg"]
model =myNet(cfg)
model.load_state_dict=check_point['state_dict']
print(cfg)
new_model = myNet(cfg)
new_model_dict=new_model.state_dict()

model={k:v for k,v in model.state_dict().items() if k in new_model_dict}
new_model_dict.update(model)
new_model.load_state_dict(new_model_dict)


transform_train=transforms.Compose([
	transforms.Resize((128,128)),
	transforms.RandomCrop((116,116)),
	transforms.RandomHorizontalFlip(),
	transforms.ToTensor(),
	transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
])

transform_val=transforms.Compose([
	transforms.Resize((116,116)),
	transforms.ToTensor(),
	transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
])

trainset=dset.ImageFolder(r'F:\safe_belt\@driver_call\call_0905_3lei\train',transform=transform_train)
print(trainset[0][0])
valset  =dset.ImageFolder(r'F:\safe_belt\@driver_call\call_0905_3lei\call_correct',transform=transform_val)
print(len(valset))
trainloader=torch.utils.data.DataLoader(trainset,batch_size=opt.batchSize,shuffle=True,num_workers=opt.num_workers)
valloader=torch.utils.data.DataLoader(valset,batch_size=opt.batchSize,shuffle=False,num_workers=opt.num_workers)


new_model.cuda()
optimizer=torch.optim.SGD(new_model.parameters(),lr=opt.lr,momentum=0.9,weight_decay=5e-4)
scheduler=StepLR(optimizer,step_size=20)
criterion=nn.CrossEntropyLoss()
criterion.cuda()

def train(epoch):
	print('\nEpoch: %d' % epoch)
	scheduler.step()
	print(scheduler.get_lr())
	new_model.train()

	# time_start = time.time()

	for batch_idx,(img,label) in enumerate(trainloader):
		# time_end = time.time()
		# print('totally cost', time_end - time_start)
		image=Variable(img.cuda())
		label=Variable(label.cuda())
		optimizer.zero_grad()
		out=new_model(image)
		loss=criterion(out,label)
		loss.backward()
		optimizer.step()
		print("Epoch:%d [%d|%d] loss:%f" %(epoch,batch_idx,len(trainloader),loss.mean()))
	# exModelName="ckp/epoth_"+str(epoch)+"_model"+".pth"
	# # torch.save(model.state_dict(),exModelName)
	# torch.save(model.state_dict(),exModelName)
def val(epoch):
	print("\nValidation Epoch: %d" %epoch)
	new_model.eval()
	total=0
	correct=0
	with torch.no_grad():
		for batch_idx,(img,label) in enumerate(valloader):
			image=Variable(img.cuda())
			label=Variable(label.cuda())
			out=new_model(image)
			_,predicted=torch.max(out.data,1)
			total+=image.size(0)
			correct+=predicted.data.eq(label.data).cpu().sum()
	accuracy=1.0*correct.numpy()/total
	print("Acc: %f "% ((1.0*correct.numpy())/total))
	exModelName = "F:\\safe_belt\\@driver_call\\call_0905_3lei\\pytorch_model\\" +str(accuracy)+"_"+"epoth_"+ str(epoch) + "_model" + ".pth"
	# torch.save(model.state_dict(),exModelName)
	torch.save(new_model.state_dict(), exModelName)

for epoch in range(opt.nepoch):
	train(epoch)
	val(epoch)
