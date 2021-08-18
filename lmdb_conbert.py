import glob
import os
import pickle
import sys

import cv2
import lmdb
import numpy as np
from tqdm import tqdm

def cv_imread(path):
    img=cv2.imdecode(np.fromfile(path,dtype=np.uint8),-1)
    return img


def main(mode):
    proj_root = r'C:\train_data\isDriver\val'
    datasets_root =r'C:\train_data\isDriver\val'
    lmdb_path = os.path.join(proj_root, 'isDriver.lmdb')
    data_path = os.path.join(datasets_root, '0')

    if mode == 'creating':
        opt = {
            'name': 'TrainSet',
            'img_folder': data_path,
            'lmdb_save_path': lmdb_path,
            'commit_interval': 100,  # After commit_interval images, lmdb commits
            'num_workers': 8,
        }
        general_image_folder(opt)
    elif mode == 'testing':
        tes_lmdb(lmdb_path, index=1)
    elif mode == "haha":
        opt = {
            'name': 'TrainSet',
            'img_folder': data_path,
            'lmdb_save_path': lmdb_path,
            'commit_interval': 100,  # After commit_interval images, lmdb commits
            'num_workers': 8,
        }
        mygeneral_image_folder(opt)


def mygeneral_image_folder(opt):
    """
    Create lmdb for general image folders
    If all the images have the same resolution, it will only store one copy of resolution info.
        Otherwise, it will store every resolution info.
    """
    txtFilePath = r"C:\train_data\isDriver\label.txt"

    img_folder = opt['img_folder']
    lmdb_save_path = opt['lmdb_save_path']
    meta_info = {'name': opt['name']}

    if not lmdb_save_path.endswith('.lmdb'):
        raise ValueError("lmdb_save_path must end with 'lmdb'.")
    if os.path.exists(lmdb_save_path):
        print('Folder [{:s}] already exists. Exit...'.format(lmdb_save_path))
        sys.exit(1)

    # read all the image paths to a list

    print('Reading image path list ...')
    all_img_list = sorted(glob.glob(os.path.join(img_folder, '*')))
    # cache the filename, 这里的文件名必须是ascii字符
    keys = []
    for img_path in all_img_list:
        keys.append(os.path.basename(img_path))

    # create lmdb environment

    # 估算大概的映射空间大小
    data_size_per_img = cv2.imread(all_img_list[0], cv2.IMREAD_UNCHANGED).nbytes
    print('data size per image is: ', data_size_per_img)
    data_size = data_size_per_img * len(all_img_list)
    env = lmdb.open(lmdb_save_path, map_size=data_size * 10)
    # map_size：
    # Maximum size database may grow to; used to size the memory mapping. If database grows larger
    # than map_size, an exception will be raised and the user must close and reopen Environment.

    # write data to lmdb

    txn = env.begin(write=True)
    resolutions = []
    tqdm_iter = tqdm(enumerate(zip(all_img_list, keys)), total=len(all_img_list), leave=False)
    for idx, (path, key) in tqdm_iter:
        tqdm_iter.set_description('Write {}'.format(key))

        key_byte = key.encode('ascii')
        data = cv2.imread(path, cv2.IMREAD_UNCHANGED)

        if data.ndim == 2:
            H, W = data.shape
            C = 1
        else:
            H, W, C = data.shape
        resolutions.append('{:d}_{:d}_{:d}'.format(C, H, W))

        txn.put(key_byte, data)
        if (idx + 1) % opt['commit_interval'] == 0:
            txn.commit()
            # commit 之后需要再次 begin
            txn = env.begin(write=True)
    txn.commit()
    env.close()
    print('Finish writing lmdb.')

    # create meta information

    # check whether all the images are the same size
    assert len(keys) == len(resolutions)
    if len(set(resolutions)) <= 1:
        meta_info['resolution'] = [resolutions[0]]
        meta_info['keys'] = keys
        print('All images have the same resolution. Simplify the meta info.')
    else:
        meta_info['resolution'] = resolutions
        meta_info['keys'] = keys
        print('Not all images have the same resolution. Save meta info for each image.')

    pickle.dump(meta_info, open(os.path.join(lmdb_save_path, 'meta_info.pkl'), "wb"))
    print('Finish creating lmdb meta info.')

def general_image_folder(opt):
    """
    Create lmdb for general image folders
    If all the images have the same resolution, it will only store one copy of resolution info.
        Otherwise, it will store every resolution info.
    """
    img_folder = opt['img_folder']
    lmdb_save_path = opt['lmdb_save_path']
    meta_info = {'name': opt['name']}

    if not lmdb_save_path.endswith('.lmdb'):
        raise ValueError("lmdb_save_path must end with 'lmdb'.")
    if os.path.exists(lmdb_save_path):
        print('Folder [{:s}] already exists. Exit...'.format(lmdb_save_path))
        sys.exit(1)

    # read all the image paths to a list

    print('Reading image path list ...')
    all_img_list = sorted(glob.glob(os.path.join(img_folder, '*')))
    # cache the filename, 这里的文件名必须是ascii字符
    keys = []
    for img_path in all_img_list:
        keys.append(os.path.basename(img_path))

    # create lmdb environment

    # 估算大概的映射空间大小
    data_size_per_img = cv2.imread(all_img_list[0], cv2.IMREAD_UNCHANGED).nbytes
    print('data size per image is: ', data_size_per_img)
    data_size = data_size_per_img * len(all_img_list)
    env = lmdb.open(lmdb_save_path, map_size=data_size * 10)
    # map_size：
    # Maximum size database may grow to; used to size the memory mapping. If database grows larger
    # than map_size, an exception will be raised and the user must close and reopen Environment.

    # write data to lmdb

    txn = env.begin(write=True)
    resolutions = []
    tqdm_iter = tqdm(enumerate(zip(all_img_list, keys)), total=len(all_img_list), leave=False)
    for idx, (path, key) in tqdm_iter:
        tqdm_iter.set_description('Write {}'.format(key))

        key_byte = key.encode('ascii')
        data = cv2.imread(path, cv2.IMREAD_UNCHANGED)

        if data.ndim == 2:
            H, W = data.shape
            C = 1
        else:
            H, W, C = data.shape
        resolutions.append('{:d}_{:d}_{:d}'.format(C, H, W))

        txn.put(key_byte, data)
        if (idx + 1) % opt['commit_interval'] == 0:
            txn.commit()
            # commit 之后需要再次 begin
            txn = env.begin(write=True)
    txn.commit()
    env.close()
    print('Finish writing lmdb.')

    # create meta information

    # check whether all the images are the same size
    assert len(keys) == len(resolutions)
    if len(set(resolutions)) <= 1:
        meta_info['resolution'] = [resolutions[0]]
        meta_info['keys'] = keys
        print('All images have the same resolution. Simplify the meta info.')
    else:
        meta_info['resolution'] = resolutions
        meta_info['keys'] = keys
        print('Not all images have the same resolution. Save meta info for each image.')

    pickle.dump(meta_info, open(os.path.join(lmdb_save_path, 'meta_info.pkl'), "wb"))
    print('Finish creating lmdb meta info.')
def tes_lmdb(dataroot, index=2):
    env = lmdb.open(dataroot, readonly=True, lock=False, readahead=False, meminit=False)
    meta_info = pickle.load(open(os.path.join(dataroot, 'meta_info.pkl'), "rb"))
    print('Name: ', meta_info['name'])
    print('Resolution: ', meta_info['resolution'])
    print('# keys: ', len(meta_info['keys']))

    # read one image
    key = meta_info['keys'][index]
    print('Reading {} for test.'.format(key))
    with env.begin(write=False) as txn:
        buf = txn.get(key.encode('ascii'))
    img_flat = np.frombuffer(buf, dtype=np.uint8)

    C, H, W = [int(s) for s in meta_info['resolution'][index].split('_')]
    img = img_flat.reshape(H, W, C)

    cv2.namedWindow('Test')
    cv2.imshow('Test', img)
    cv2.waitKeyEx()


if __name__ == "__main__":
    # mode = creating or testing
    main(mode='creating')
