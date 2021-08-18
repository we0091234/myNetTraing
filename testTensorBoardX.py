import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import *
import torchvision.transforms as transforms
import cvtorchvision.cvtransforms as cvTransforms
import torchvision.datasets as dset
from tensorboardX import SummaryWriter
from myNet import myNet


x =torch.rand(1,3,116,116)
model = myNet()
with SummaryWriter(comment="net1") as w:
    w.add_graph(model,(x,))
