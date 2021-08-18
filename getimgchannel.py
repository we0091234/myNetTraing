import cv2
import os
import numpy as np
def cv_imread(path):
    img=cv2.imdecode(np.fromfile(path,dtype=np.uint8),-1)
    return img
def allFilePath(rootPath,allFIleList):
    fileList = os.listdir(rootPath)
    for temp in fileList:
        if os.path.isfile(os.path.join(rootPath,temp)):
            allFIleList.append(os.path.join(rootPath,temp))
        else:
            allFilePath(os.path.join(rootPath,temp),allFIleList)
jpgFile = r"C:\Train_data\@changpengzhuagnheng\@changpeng\train"

fileList =[]
allFilePath(jpgFile,fileList)
for temp in fileList:
    if temp.endswith(".jpg"):
        img =cv_imread(temp)
        if img.shape[2]!=3:
            print(temp)

