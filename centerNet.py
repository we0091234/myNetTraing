import argparse
import numpy as np
import os
from torchviz import make_dot, make_dot_from_trace
from graphviz import Digraph

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
from myNet import  myNet
import torchvision.datasets as dset
from dlav0 import dla34

model =dla34(False)
x =torch.randn(3,3,256,256)
print(model)
print(model(x).shape)
g = make_dot(model(x))
g.view()
