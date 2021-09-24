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
import math
from myNet import myNet
import cv2
try:
    from nvidia.dali.plugin.pytorch import DALIClassificationIterator
    from nvidia.dali.pipeline import Pipeline
    import nvidia.dali.ops as ops
    import nvidia.dali.types as types
except ImportError:
    raise ImportError("Please install DALI from https://www.github.com/NVIDIA/DALI to run this example.")
def cv_imread(path):
    img=cv2.imdecode(np.fromfile(path,dtype=np.uint8),-1)
    return img

class HybridTrainPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, crop,
                 shard_id, num_shards, dali_cpu=False):
        super(HybridTrainPipe, self).__init__(batch_size,
                                              num_threads,
                                              device_id,
                                              seed=12 + device_id)
        self.input = ops.FileReader(file_root=data_dir,
                                    shard_id=0,
                                    num_shards=1,
                                    random_shuffle=True,
                                    pad_last_batch=True)
        #let user decide which pipeline works him bets for RN version he runs
        dali_device = 'cpu' if dali_cpu else 'gpu'
        decoder_device = 'cpu' if dali_cpu else 'mixed'
        # This padding sets the size of the internal nvJPEG buffers to be able to handle all images from full-sized ImageNet
        # without additional reallocations
        device_memory_padding = 211025920 if decoder_device == 'mixed' else 0
        host_memory_padding = 140544512 if decoder_device == 'mixed' else 0
        self.decode = ops.ImageDecoderRandomCrop(device=decoder_device, output_type=types.BGR,
                                                 device_memory_padding=device_memory_padding,
                                                 host_memory_padding=host_memory_padding,
                                                 random_aspect_ratio=[0.8, 1.25],
                                                 random_area=[0.1, 1.0],
                                                 num_attempts=100)
        self.res = ops.Resize(device=dali_device,
                              resize_x=crop,
                              resize_y=crop,
                              interp_type=types.INTERP_TRIANGULAR)
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            crop=(crop, crop),
                                            image_type=types.BGR,
                                            mean=[99.95327338 ,96.27925874, 86.54154894],
                                            std=[1,1,1]
											#   mean=mean,
                                            # std=[1,1,1])
		)
        self.coin = ops.CoinFlip(probability=0.5)
        print('DALI "{0}" variant'.format(dali_device))

    def define_graph(self):
        rng = self.coin()
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        images = self.res(images)
        output = self.cmnp(images.gpu(), mirror=rng)
        return [output, self.labels]


class HybridValPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, crop,
                 size, shard_id, num_shards):
        super(HybridValPipe, self).__init__(batch_size,
                                           num_threads,
                                            device_id,
                                            seed=12 + device_id)
        self.input = ops.FileReader(file_root=data_dir,
                                    shard_id=0,
                                    num_shards=1,
                                    random_shuffle=False,
                                    pad_last_batch=True)
        self.decode = ops.ImageDecoder(device="mixed", output_type=types.BGR)
        self.res = ops.Resize(device="gpu",
                              resize_shorter=size,
                              interp_type=types.INTERP_TRIANGULAR)
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            crop=(crop, crop),
                                            image_type=types.BGR,
                                             mean=mean,
                                            std=[1,1,1])

    def define_graph(self):
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        images = self.res(images)
        output = self.cmnp(images)
        # output=images
        return [output, self.labels]

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
parser.add_argument('--nepoch',type=int,default=120)
parser.add_argument('--lr',type=float,default=0.025)
parser.add_argument('--gpu',type=str,default='0')
opt=parser.parse_args()
# print(opt)
os.environ["CUDA_VISIBLE_DEVICES"]=opt.gpu
device=torch.device("cuda")
torch.backends.cudnn.benchmark=True
# MEAN_NPY = r'C:\Train_data\@changpengzhuagnheng\@penzi\vehicle.npy'
MEAN_NPY = r'/home/xiaolei/train_data/myNetTraing/meanFile/VehicleDriverGeneral.npy'
# 'G:\driver_shenzhen\@new\VehicleDriverGeneral.npy'
mean_npy = np.load(MEAN_NPY)
mean = mean_npy.mean(1).mean(1)
print(mean)
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
	cvTransforms.NormalizeCaffe( [99.95327338 ,96.27925874, 86.54154894]),
])

# def fix_bn(m):
#     classname = m.__class__.__name__
#     if  classname.find('BatchNorm') != -1:
#         if m.num_features==32:
#            m.eval()





def train(epoch,scheduler,model,train_loader,criterion,optimizer):
	print('\nEpoch: %d' % epoch)
	scheduler.step()
	print(scheduler.get_lr())
	model.train()
	# model.apply(fix_bn)

	# time_start = time.time()
	train_loader_len = int(math.ceil(train_loader._size / 512))
	# tmp = list(enumerate(train_loader))
	# tmp = list(enumerate(train_loader))
	for batch_idx, data in enumerate(train_loader):
		# time_end = time.time()
		# print('totally cost', time_end - time_start)

		image = data[0]["data"]
		label = data[0]["label"].squeeze(-1).long()


		# image=Variable(img.cuda())
		# label=Variable(label.cuda())
		optimizer.zero_grad()
		out=model(image)
		loss=criterion(out,label)
		loss.backward()
		optimizer.step()
		if batch_idx%10==0:
			print("Epoch:%d [%d|%d] loss:%f lr:%s" %(epoch,batch_idx,train_loader_len,loss.mean(),scheduler.get_lr()))
	# exModelName="ckp/epoth_"+str(epoch)+"_model"+".pth"
	# # torch.save(model.state_dict(),exModelName)
	# torch.save(model.state_dict(),exModelName)
def val(epoch,model,valloader):
	print("\nValidation Epoch: %d" %epoch)
	model.eval()
	total=0
	correct=0
	with torch.no_grad():
		for batch_idx,(img,label) in enumerate(valloader):
			# image = data[0]["data"]
			# label = data[0]["label"].squeeze().cuda().long()
			image=Variable(img.cuda())
			label=Variable(label.cuda())
			out=model(image)
			_,predicted=torch.max(out.data,1)
			total+=image.size(0)
			correct+=predicted.data.eq(label.data).cpu().sum()
	accuracy=1.0*correct.numpy()/total
	print("Acc: %f "% ((1.0*correct.numpy())/total))
	exModelName = r"/home/xiaolei/train_data/myNetTraing/model_driver/" +str(format(accuracy,'.6f'))+"_"+"epoth_"+ str(epoch) + "_model" + ".pth.tar"
	# torch.save(model.state_dict(),exModelName)
	torch.save({'cfg': myCfg, 'state_dict': model.state_dict()}, exModelName)

if __name__ == '__main__':

	pipe = HybridTrainPipe(batch_size=512,
							num_threads=4,
							device_id=0,
							data_dir='/home/xiaolei/train_data/myNetTraing/datasets/driverDall/train',
							crop=128,
							dali_cpu=False,
							shard_id=0,
							num_shards=1)
	pipe.build()
	train_loader = DALIClassificationIterator(pipe, reader_name="Reader", fill_last_batch=False)

	# pipe = create_dali_pipeline(batch_size=32,
	# 							num_threads=4,
	# 							device_id=0,
	# 							seed=12 + 0,
	# 							data_dir='/home/cxl/pytorchTrain/trainData/DrivalCall/val',
	# 							crop=128,
	# 							size=128,
	# 							dali_cpu=False,
	# 							shard_id=0,
	# 							num_shards=1,
	# 							is_training=True)
	# pipe.build()
	# val_loader = DALIClassificationIterator(pipe, reader_name="Reader", last_batch_policy=LastBatchPolicy.PARTIAL)
	# val_loader = DALIClassificationIterator(pipe, reader_name="Reader", last_batch_policy=LastBatchPolicy.PARTIAL)
	# trainset = dset.ImageFolder(r'/home/cxl/pytorchTrain/trainData/DrivalCall/train', transform=transform_train,loader=cv_imread)
	# print(trainset[0][0])
	valset = dset.ImageFolder(r'/home/xiaolei/train_data/myNetTraing/datasets/driverDall/val', transform=transform_val,loader=cv_imread)
	# # print(len(valset))
	# # trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.batchSize, shuffle=True,
	# # 										  num_workers=opt.num_workers)
	valloader = torch.utils.data.DataLoader(valset, batch_size=opt.batchSize, shuffle=False,
											num_workers=opt.num_workers)

	# pipe = HybridValPipe(batch_size=32,
	# 						num_threads=4,
	# 						device_id=0,
	# 						data_dir="/home/cxl/pytorchTrain/trainData/DrivalCall/new/val",
	# 						crop=128,
	# 						size=128,
	# 						shard_id=0,
	# 						num_shards=1)
	# pipe.build()
	# val_loader = DALIClassificationIterator(pipe, reader_name="Reader", fill_last_batch=False)

	myCfg = [32, 'M', 64, 'M', 96, 'M', 128, 'M', 192, 'M', 256]
	model = myNet(num_classes=3,cfg=myCfg)
	model.cuda()
	optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9, weight_decay=5e-4)
	# scheduler=StepLR(optimizer,step_size=20)
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(opt.nepoch))
	# criterion=nn.CrossEntropyLoss()
	criterion = CrossEntropyLabelSmooth(3)
	criterion.cuda()

	for epoch in range(opt.nepoch):
		train(epoch,scheduler,model,train_loader,criterion,optimizer)
		
		val(epoch,model,valloader)
		train_loader.reset()
		# val_loader.reset()

