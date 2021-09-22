import os
import os.path as osp
import os, sys
import os.path as osp
# from PIL import Image
import six
import string
import numpy as np
import lmdb
import pickle
import umsgpack
import tqdm
import pyarrow as pa
from os.path import basename
import argparse
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder
from torchvision import transforms, datasets
import cv2
import cvtorchvision.cvtransforms as cvTransforms


class ImageFolderWithPaths(datasets.ImageFolder):
    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


def loads_pyarrow(buf):
    """
    Args:
        buf: the output of `dumps`.
    """
    return pa.deserialize(buf)


def read_txt(fname):
    map = {}
    with open(fname) as f:
        content = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]
    for line in content:
        img, idx = line.split(" ")
        map[img] = idx
    return map

def read_txt1(fname):
    map = []
    with open(fname) as f:
        content = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]
    for line in content:
        img, idx = line.split(" ")
        map.append(img)
    return map


class ImageFolderLMDB(data.Dataset):
    def __init__(self, db_path, transform=None, target_transform=None):
        self.db_path = db_path
        self.env = lmdb.open(db_path, subdir=osp.isdir(db_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            # self.length = txn.stat()['entries'] - 1
            self.length = loads_pyarrow(txn.get(b'__len__'))
            # self.keys = umsgpack.unpackb(txn.get(b'__keys__'))
            self.keys = loads_pyarrow(txn.get(b'__keys__'))

        self.transform = transform
        self.target_transform = target_transform
        # map_path = db_path[:-5] + "_images_idx.txt"
        # self.img2idx = read_txt1(map_path)

    def __getitem__(self, index):
        img, target = None, None
        # imgName=self.img2idx[index]
        # target=int(imgName.split("-")[-1].split(".")[0])
        env = self.env
        with env.begin(write=False) as txn:
            # print("key", self.keys[index].decode("ascii"))
            byteflow = txn.get(self.keys[index])

        unpacked = loads_pyarrow(byteflow)

        imgbuf = unpacked
        buf = six.BytesIO()
        buf.write(imgbuf[0])
        buf.seek(0)
        import numpy as np
        img=imgbuf[0]
        # img = Image.open(buf).convert('RGB')
        # img.save("img.jpg")
        if self.transform is not None:
            img = self.transform(img)
        im2arr = np.array(img)
        # print(im2arr.shape)

        if self.target_transform is not None:
            target = self.target_transform(target)

        # return img, target
        return im2arr,imgbuf[1]

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'


def raw_reader(path):
    with open(path, 'rb') as f:
        bin_data = f.read()
    return bin_data

def cv_imread(path):
    img=cv2.imdecode(np.fromfile(path,dtype=np.uint8),-1)
    return img

def dumps_pyarrow(obj):
    """
    Serialize an object.
    Returns:
        Implementation-dependent bytes-like object
    """
    return pa.serialize(obj).to_buffer()


def folder2lmdb(dpath,width,height ,lmdbPath="train", mapeSize=2,write_frequency=1000,resizeState=True):
    all_imgpath = []
    all_idxs = []
    directory = dpath
    print("Loading dataset from %s" % directory)
    dataset = ImageFolderWithPaths(directory, loader=cv_imread)
    ###adddd
    print(len(dataset))
    imgori,_,_=dataset[0]
    if(resizeState==False):
        imgoriSize = imgori.nbytes
    else:
        imgoriSize = cv2.resize(imgori, (width, height), interpolation=cv2.INTER_LINEAR).nbytes
    data_size = imgoriSize * len(dataset)
    ######ddddd
    data_loader = DataLoader(dataset, num_workers=4, collate_fn=lambda x: x)
    lmdb_path =lmdbPath
    isdir = os.path.isdir(lmdb_path)

    print("Generate LMDB to %s" % lmdb_path)
    db = lmdb.open(lmdb_path, subdir=isdir,
                   map_size=data_size*mapeSize, readonly=False,
                   meminit=False, map_async=True)

    txn = db.begin(write=True)
    for idx, data in enumerate(data_loader):
        # print(type(data), data)
        # image, label = data[0]
        image, label, imgpath = data[0]
        ####aaddddd
        if(resizeState):
            image=cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
        ####adddd
        # print(image.shape)
        imgpath = basename(imgpath)
        all_imgpath.append(imgpath)
        all_idxs.append(idx)
        imageLabel=(image,label)
        txn.put(u'{}'.format(idx).encode('ascii'), dumps_pyarrow(imageLabel))
        # txn.put(u'{}'.format(imgpath).encode('ascii'), dumps_pyarrow(image))
        if idx % write_frequency == 0:
            print("[%d/%d]" % (idx, len(data_loader)))
            txn.commit()
            txn = db.begin(write=True)

    # finish iterating through dataset
    txn.commit()
    keys = [u'{}'.format(k).encode('ascii') for k in range(idx + 1)]
    with db.begin(write=True) as txn:
        txn.put(b'__keys__', dumps_pyarrow(keys))
        txn.put(b'__len__', dumps_pyarrow(len(keys)))

    print("Flushing database ...")
    db.sync()
    db.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lmdbPath', type=str, default=r'/home/xiaolei/train_data/myNetTraing/datasets/datasets/pedestrain/gender/trainAug.lmdb')
    parser.add_argument('--imgSizeW', type=int, default=128)
    parser.add_argument('--imgSizeH', type=int, default=128)
    parser.add_argument('--picPath', type=str, default=r'/home/xiaolei/train_data/myNetTraing/datasets/datasets/pedestrain/gender/train_aug')
    parser.add_argument('--mapSize', type=float, default=10)
    opt = parser.parse_args()
    print (opt)
    folder2lmdb(opt.picPath,opt.imgSizeW,opt.imgSizeH,lmdbPath=opt.lmdbPath,mapeSize=opt.mapSize,resizeState=False)
