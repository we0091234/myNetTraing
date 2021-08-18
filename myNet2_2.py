import math

import torch
import torch.nn as nn
from torch.autograd import Variable


__all__ = ['myNet']

# defaultcfg = {
#     11 : [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
#     13 : [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
#     16 : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
#     19 : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
# }

myCfg = [32,'M',64,'M',96,'M',128,'M',192,'M',256]
class myNet(nn.Module):
    def __init__(self, cfg=None, num_classes=[6]):
        super(myNet, self).__init__()
        if cfg is None:
            cfg = myCfg

        self.feature = self.make_layers(cfg[0:4], True)
        self.feature2 = self.make_layers(cfg[4:11], True)
        
        self.classifier1 = nn.Sequential(
            nn.Linear(cfg[10], num_classes[0]),
            nn.BatchNorm1d(num_classes[0])
        )

        # self.feature3 = self.make_layers(cfg[11:18], True)
        # self.classifier2 = nn.Sequential(
        #     nn.Linear(cfg[17], num_classes[1]),
        #     nn.BatchNorm1d(num_classes[1])
        # )

    def make_layers(self, cfg, batch_norm=False):
        layers = []

        in_channels = 3
        if len(cfg) > 4:
            in_channels = 64

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
        x1 = self.feature(x)

        x2 = self.feature2(x1)
        x3 = nn.AvgPool2d(kernel_size=3, stride=1)(x2)
        x4 = x3.view(x3.size(0), -1)
        y1 = self.classifier1(x4)

        # x2_2 = self.feature3(x1)
        # x3_2 = nn.AvgPool2d(kernel_size=3, stride=1)(x2_2)
        # x4_2 = x3_2.view(x3_2.size(0), -1)
        # y2 = self.classifier2(x4_2)

        return y1

    def backone_params(self):
        params = self.feature.parameters()
        return params

    def finetune_params(self):
        params = []
        for param in self.feature2.parameters():
            params.append(param)
        for param in self.classifier1.parameters():
            params.append(param)
        return params

if __name__ == '__main__':
    net = myNet()
    # x = Variable(torch.FloatTensor(16, 3, 128, 128))
    # y = net(x)
    # print('y1: ', y.data.shape)
    print(net)