# -*- coding: utf-8 -*
import os
import cv2
import numpy as np
path1=r'H:\@chepai\@changpengZHuangheng\zhangpengData\V2_Aug_PenziClas\2\111'
path2=r'H:\@chepai\@changpengZHuangheng\zhangpengData\V2_Aug_PenziClas\2\0_low_resize'
filelist1=os.listdir(path1)
def allFilePath(rootPath,allFIleList):
    fileList = os.listdir(rootPath)
    for temp in fileList:
        if os.path.isfile(os.path.join(rootPath,temp)):
            allFIleList.append(os.path.join(rootPath,temp))
        else:
            allFilePath(os.path.join(rootPath,temp),allFIleList)
num=0
for file1 in filelist1:
        picNum = 0
        data_path1=os.path.join(path1,file1)
        print(num)
        num+=1
        filelist2=[]
        allFilePath(data_path1,filelist2)
        npath1=os.path.join(path2,file1)
        if not os.path.isdir(npath1):
            os.mkdir(npath1)
        for pic in filelist2:
                picNum+=1
                pic_path=pic
                picName = pic.split("\\")[-1]
                picName1 = str(picNum)+"_"+picName
                save_path=os.path.join(npath1,picName1)
                #img=Image.open(pic_path)
                #src=img.resize((128,128),Image.ANTIALIAS)
                #src.save(save_path)
                ##############
                #img=cv2.imread(pic_path)
                #src=cv2.resize(img,(128,128),interpolation=cv2.INTER_LINEAR)
                #cv2.imwrite(save_path,src)
                ########
                num+=1
                if num%1000==0:
                    print(num)
                img=cv2.imdecode(np.fromfile(pic_path,dtype=np.uint8),-1)
                src=cv2.resize(img,(140,140),interpolation=cv2.INTER_LINEAR)
                cv2.imencode('.jpg', src)[1].tofile(save_path)
