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
NUM_IMG=50
def main():
    path=r"/home/lxk/ZHP/data/ReIDData/ReID_lowerAndupper/ReID_upper/YTA"
    # path=r"/home/lxk/ZHP/data/ReIDData_Aug/YT3"  #使用绝对路径
    save_path=r'/home/data_ssd/ZHP/ReID_upper/YTA'
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
            if num>0 and num <NUM_IMG:
                num_of_samples = int(NUM_IMG-num)
                p = Augmentor.Pipeline(roof,new_path)
                p.resize(probability=1, width=width, height=height, resample_filter="BICUBIC")
                p.random_erasing(probability=0.4, rectangle_area=0.3)
                p.skew_corner(probability=0.5, magnitude=0.3)
                p.process()
                p1 = Augmentor.Pipeline(roof,new_path)
                p1.resize(probability=1, width=width, height=height, resample_filter="BICUBIC")
                # p1.crop_random(probability=0.5, percentage_area=0.9)
                # p1.flip_left_right(probability=0.5)
                p1.random_erasing(probability=0.4, rectangle_area=0.3)
                p1.rotate(probability=0.5,max_left_rotation=5,max_right_rotation=5)
                # p1.random_color(probability=0.3,min_factor=5,max_factor=10)
                # p1.random_brightness(probability=0.3,min_factor=0.3,max_factor=3)
                p1.skew_corner(probability=0.5, magnitude=0.1)
                p1.sample(num_of_samples,multi_threaded=True)
            else:
                p=Augmentor.Pipeline(roof,new_path)
                p.resize(probability=1, width=width, height=height, resample_filter="BICUBIC")
                p.random_erasing(probability=0.4, rectangle_area=0.3)
                # p.flip_left_right(probability=0.5)
                p.rotate(probability=0.5,max_left_rotation=5,max_right_rotation=5)
                p.skew_corner(probability=0.5, magnitude=0.1)
                # p.random_color(probability=0.3,min_factor=5,max_factor=10)
                # p.random_brightness(probability=0.3,min_factor=0.3,max_factor=3)
                p.process()
            flag+=1
if __name__ == '__main__':
    main()