import math

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models


__all__ = ['myNet','myResNet18']

# defaultcfg = {
#     11 : [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
#     13 : [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
#     16 : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
#     19 : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
# }
myCfg = [32,'M',64,'M',96,'M',128,'M',192,'M',256]
class myNet(nn.Module):
    def __init__(self,cfg=None,num_classes=3):
        super(myNet, self).__init__()
        if cfg is None:
            cfg = myCfg
        self.feature = self.make_layers(cfg, True)
        self.classifier = nn.Linear(cfg[-1], num_classes)
    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for i in range(len(cfg)):
            if i == 0:
                conv2d =nn.Conv2d(in_channels, cfg[i], kernel_size=5,stride =1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(cfg[i]), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = cfg[i]
            else :
                if cfg[i] == 'M':
                    layers += [nn.MaxPool2d(kernel_size=3, stride=2,ceil_mode=True)]
                else:
                    conv2d = nn.Conv2d(in_channels, cfg[i], kernel_size=3, padding=1,stride =1)
                    if batch_norm:
                        layers += [conv2d, nn.BatchNorm2d(cfg[i]), nn.ReLU(inplace=True)]
                    else:
                        layers += [conv2d, nn.ReLU(inplace=True)]
                    in_channels = cfg[i]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.feature(x)
        x = nn.AvgPool2d(kernel_size=3, stride=1)(x)
        x = x.view(x.size(0), -1)
        y = self.classifier(x)
        return y

class myResNet18(nn.Module):
    def __init__(self,num_classes=1000):
        super(myResNet18,self).__init__()
        model_ft = models.resnet18(pretrained=True)
        self.model =model_ft
        self.model.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1,ceil_mode=True)
        self.model.averagePool = nn.AvgPool2d((5,5),stride=1,ceil_mode=True)
        self.cls=nn.Linear(512,num_classes)

    def forward(self,x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.averagePool(x)
        x = x.view(x.size(0), -1)
        x = self.cls(x)

        return x
if __name__ == '__main__':
    net = myResNet18()
    x = Variable(torch.FloatTensor(16, 3, 128, 128))
    y = net(x)
    print(y.data.shape)
    # print(net)