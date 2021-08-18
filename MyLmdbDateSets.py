from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch
import cv2
import cvtorchvision.cvtransforms as cvTransforms
import numpy as np
import glob
import os
import pickle
import sys
import lmdb
from tqdm import tqdm
def cv_imread(path):
    img=cv2.imdecode(np.fromfile(path,dtype=np.uint8),-1)
    return img


__all__ = ['MyLmdbDataSets']
class MyLmdbDataSets(Dataset):
    def __init__(self,txtPath,transform=None,target_transform=None):
        self.imgfile=[]
        self.transform = transform
        f=open(txtPath,"r")
        while True:
            line = f.readline().strip("\n")
            if not line:
                break
            self.imgfile.append( (line.split(" ")[0],line.split(" ")[1]))
        f.close()
    def __getitem__(self, item):
        imgPath,label=self.imgfile[item]
        img=cv_imread(imgPath)
        if self.transform is not None:
            img=self.transform(img)
        # label=self.labelfile[item]
        return  img,int(label)
    def __len__(self):
        return len(self.imgfile)

# transform_train = cvTransforms.Compose([
#             cvTransforms.Resize((140, 140)),
#             cvTransforms.RandomCrop((128, 128)),
#             cvTransforms.RandomHorizontalFlip(),  # 镜像
#             cvTransforms.ToTensorNoDiv(),  # caffe中训练没有除以255所以 不除以255
#             # cvTransforms.NormalizeCaffe(mean)  # caffe只用减去均值
#         ])
#
if __name__ == '__main__':
    data=MyLmdbDataSets(r"C:\train_data\isDriver\label.txt")
    for file in data:
        cv2.imshow("haha", file[0])
        cv2.waitKey(0)

