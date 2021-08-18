import cv2
import numpy as np
import os

def allFilePath(rootPath,allFIleList):
    fileList = os.listdir(rootPath)
    for temp in fileList:
        if os.path.isfile(os.path.join(rootPath,temp)):
            allFIleList.append(os.path.join(rootPath,temp))
        else:
            allFilePath(os.path.join(rootPath,temp),allFIleList)

fileList =[]
rootPath =r"I:\datasets\danger_detection\upBody_cloth\train"
allFilePath(rootPath,fileList)
for  temp in fileList:
    img = cv2.imread(temp)
    size=os.path.getsize(temp)
    if size<1:
        os.remove(temp)
        print(temp)