import argparse
import numpy as np
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
import cvtorchvision.cvtransforms as cvTransforms
from myNet import  myNet
import torchvision.datasets as dset
import cv2


# Prune settings
# parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR prune')
# parser.add_argument('--dataset', type=str, default='cifar100',
#                     help='training dataset (default: cifar10)')
# parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
#                     help='input batch size for testing (default: 256)')
# parser.add_argument('--no-cuda', action='store_true', default=False,
#                     help='disables CUDA training')
# parser.add_argument('--depth', type=int, default=19,
#                     help='depth of the vgg')
# parser.add_argument('--percent', type=float, default=0.3,
#                     help='scale sparse rate (default: 0.5)')
# parser.add_argument('--model', default='', type=str, metavar='PATH',
#                     help='path to the model (default: none)')
# parser.add_argument('--save', default='', type=str, metavar='PATH',
#                     help='path to save pruned model (default: none)')
cuda=1
percent=0.3
MEAN_NPY = r'H:\@pedestrain_datasets\sex\mean.npy'
mean_npy = np.load(MEAN_NPY)
mean = mean_npy.mean(1).mean(1)

transform_val=cvTransforms.Compose([
	cvTransforms.Resize((128,128)),
	cvTransforms.ToTensorNoDiv(),
	cvTransforms.NormalizeCaffe(mean),
])
saveFolder =r"I:\BaiduNetdiskDownload\dogsAndCats\Kaggle-Dogs_vs_Cats_PyTorch-master\pruneModel"
model = myNet(num_classes=4)
model.cuda()

model.load_state_dict(torch.load(r'D:\@linux_share\coatText0.5\epoth_0.9444444444444444_epoth_28_model.pth.tar'))

print(model)
total = 0
for m in model.modules():
    if isinstance(m, nn.BatchNorm2d):
        total += m.weight.data.shape[0]

bn = torch.zeros(total)
index = 0
for m in model.modules():
    if isinstance(m, nn.BatchNorm2d):
        size = m.weight.data.shape[0]
        bn[index:(index+size)] = m.weight.data.abs().clone()
        index += size

y, i = torch.sort(bn)
thre_index = int(total * percent)
thre = y[thre_index]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
thre =thre.to(device)
pruned = 0
cfg = []
cfg_mask = []
for k, m in enumerate(model.modules()):
    if isinstance(m, nn.BatchNorm2d):
        weight_copy = m.weight.data.abs().clone()
        # print(m.weight.data.shape)
        # print(weight_copy.device)
        # weight_copy=weight_copy.to(device)
        mask = weight_copy.gt(thre).float().cuda()
        # mask = weight_copy.gt(thre).float()
        pruned = pruned + mask.shape[0] - torch.sum(mask)
        m.weight.data.mul_(mask)
        m.bias.data.mul_(mask)
        cfg.append(int(torch.sum(mask)))
        cfg_mask.append(mask.clone())
        print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
            format(k, mask.shape[0], int(torch.sum(mask))))
    elif isinstance(m, nn.MaxPool2d):
        cfg.append('M')

pruned_ratio = pruned/total

print('Pre-processing Successful!')

# simple test model after Pre-processing prune (simple set BN scales to zeros)
def test(model):
    valset = dset.ImageFolder(r'M:\GetSmallPIc\4', transform=transform_val)
    valloader = torch.utils.data.DataLoader(valset, batch_size=32, shuffle=False,
                                            num_workers=0)
    # print("\nValidation Epoch: %d" % epoch)
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (img, label) in enumerate(valloader):
            image = Variable(img.cuda())
            label = Variable(label.cuda())
            out = model(image)
            _, predicted = torch.max(out.data, 1)
            total += image.size(0)
            correct += predicted.data.eq(label.data).cpu().sum()
    accuracy = 1.0 * correct.numpy() / total
    print("Acc: %f " % ((1.0 * correct.numpy()) / total))

acc = test(model)

# Make real prune
print(cfg)
newmodel = myNet( cfg=cfg)
if cuda:
    newmodel.cuda()

num_parameters = sum([param.nelement() for param in newmodel.parameters()])
# savepath = os.path.join(args.save, "prune.txt")
# with open(savepath, "w") as fp:
#     fp.write("Configuration: \n"+str(cfg)+"\n")
#     fp.write("Number of parameters: \n"+str(num_parameters)+"\n")
#     fp.write("Test accuracy: \n"+str(acc))

layer_id_in_cfg = 0
start_mask = torch.ones(3)
end_mask = cfg_mask[layer_id_in_cfg]
for [m0, m1] in zip(model.modules(), newmodel.modules()):
    if isinstance(m0, nn.BatchNorm2d):
        idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
        if idx1.size == 1:
            idx1 = np.resize(idx1,(1,))
        m1.weight.data = m0.weight.data[idx1.tolist()].clone()
        m1.bias.data = m0.bias.data[idx1.tolist()].clone()
        m1.running_mean = m0.running_mean[idx1.tolist()].clone()
        m1.running_var = m0.running_var[idx1.tolist()].clone()
        layer_id_in_cfg += 1
        start_mask = end_mask.clone()
        if layer_id_in_cfg < len(cfg_mask):  # do not change in Final FC
            end_mask = cfg_mask[layer_id_in_cfg]
    elif isinstance(m0, nn.Conv2d):
        idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
        idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
        print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
        if idx0.size == 1:
            idx0 = np.resize(idx0, (1,))
        if idx1.size == 1:
            idx1 = np.resize(idx1, (1,))
        w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
        w1 = w1[idx1.tolist(), :, :, :].clone()
        m1.weight.data = w1.clone()
    elif isinstance(m0, nn.Linear):
        idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
        if idx0.size == 1:
            idx0 = np.resize(idx0, (1,))
        m1.weight.data = m0.weight.data[:, idx0].clone()
        m1.bias.data = m0.bias.data.clone()

torch.save({'cfg': cfg, 'state_dict': newmodel.state_dict()}, os.path.join(saveFolder, 'havepruned.pth.tar'))

print(newmodel)
model = newmodel
test(model)