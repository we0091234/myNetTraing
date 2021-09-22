# -*- coding: utf-8 -*-
import random
import os
import shutil
from multiprocessing import Pool as ProcessPool
from multiprocessing.dummy import Pool as ThreadPool
import Augmentor
import sys
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
def main():
    path=r"/home/lxk/ZHP/data/ReIDData/ReID_lowerAndupper/ReID_upper/YTC"
    # path=r"/home/lxk/ZHP/data/ReIDData_Aug/YT3"  #使用绝对路径
    save_path=r'/home/data_ssd/ZHP/ReID_upper/YTC'
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    flag=0
    width = 96
    height =96
    for roof, d, filelist in os.walk(path):
        if not d and  filelist:
            num = len(filelist)
            print()
            print('***',flag,'***',roof)
            label_name=os.path.basename(roof)
            new_path=save_path+'/'+label_name
            if not os.path.exists(new_path):
                os.makedirs(new_path)
            if num>0 and num <10:
                num_of_samples = 10 - num
                p = Augmentor.Pipeline(roof,new_path)
                p.resize(probability=1, width=width, height=height, resample_filter="BICUBIC")
                p.process()
                p1 = Augmentor.Pipeline(roof,new_path)
                p1.resize(probability=1, width=width, height=height, resample_filter="BICUBIC")
                p.skew_corner(probability=1, magnitude=0.3)
                p1.sample(num_of_samples,multi_threaded=True)
            elif num >=10 and num <=50:
                p = Augmentor.Pipeline(roof,new_path)
                p.resize(probability=1, width=width, height=height, resample_filter="BICUBIC")
                p.process()
            else:
                num_of_samples = 50
                p = Augmentor.Pipeline(roof,new_path)
                p.resize(probability=1, width=width, height=height, resample_filter="BICUBIC")
                p.sample(num_of_samples,multi_threaded=True)

            flag+=1
if __name__ == '__main__':
    main()