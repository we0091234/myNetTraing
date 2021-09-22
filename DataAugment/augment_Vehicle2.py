#-*- coding:utf-8 -*-

import random
import os
import shutil
from multiprocessing import Pool as ProcessPool
from multiprocessing.dummy import Pool as ThreadPool
import Augmentor
import sys

def main():
    path=r"/home/lxk/ZHP/data/VeIDData/V2"
    save_path=r"/home/lxk/ZHP/data/VeIDData_Aug/V2"  #使用绝对路径
    g = os.walk(path)
    flag=0
    NUM_IMG=20
    for paths, d, filelist in g:
        if not filelist:
            continue
        print(flag,paths)
        if d:
            os.system('pause')
        num = len(filelist)
        label_name=os.path.basename(paths)
        new_path=save_path+'/'+label_name
        p = Augmentor.Pipeline(paths,new_path)
        
        p.random_erasing(probability=0.5, rectangle_area=0.3)
        p.flip_left_right(probability=0.5)
        p.skew_corner(probability=0.5, magnitude=0.1)
        p.rotate(probability=0.5, max_left_rotation=5, max_right_rotation=5)
        p.crop_random(probability=0.5, percentage_area=0.9)
        p.resize(probability=1, width=128, height=128, resample_filter="BICUBIC")
        p.process()
        if num<NUM_IMG and num>0:
            num_of_samples = int(NUM_IMG-num)
            p.sample(num_of_samples,multi_threaded=True)
        flag+=1
if __name__ == '__main__':
    main()