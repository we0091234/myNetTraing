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
import adabound
from lr_scheduler import LRScheduler

# olddict = torch.load(r"H:\daTUtest\zxc_st\@cpk_model\MODEL128\Pants\0.9566473988439307_epoth_41_model.pth")
# with open("new1.txt","w") as pf:
# 	for k,v in olddict.items():
# 		pf.write("\n"+k + "\n")
# 		a = v.cpu().numpy()
# 		b = a.ravel()
# 		for i in b:
# 			pf.write(str(i) + " ")
			# pf.write("\n\n")
# checkPoint=torch.load(r"K:\PytorchToCaffe-master\@caffemodel\prune_128duiqi\epoth_0.9299867899603699_epoth_326_model.pth.tar")
# newdict = checkPoint["state_dict"]
# cfg = checkPoint["cfg"]
# print(cfg)
MEAN_NPY = r'F:\safe_belt\@driver_call\call_0905_3lei\mean.npy'
mean_npy = np.load(MEAN_NPY)
mean = mean_npy.mean(1).mean(1)
print(mean)
# with open("new2.txt","w") as pf:
# 	for k,v in newdict.items():
# 		pf.write("\n"+k + "\n")
# 		a = v.cpu().numpy()
# 		b = a.ravel()
# 		for i in b:
# 			pf.write(str(i) + " ")