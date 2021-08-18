import os
import numpy as np
import cv2
import shutil
import random
def allFilePath(rootPath,allFIleList):
    fileList = os.listdir(rootPath)
    for temp in fileList:
        if os.path.isfile(os.path.join(rootPath,temp)):
            allFIleList.append(os.path.join(rootPath,temp))
        else:
            allFilePath(os.path.join(rootPath,temp),allFIleList)
def cv_imread(path):
    img=cv2.imdecode(np.fromfile(path,dtype=np.uint8),-1)
    return img

rootFile = r"J:\@YueB\@fenlei\allfile"
saveFile =r"J:\@YueB\@fenlei\val"

folderList = os.listdir(rootFile)
for folder in folderList:
    folderPath = os.path.join(rootFile,folder)
    fileList =[]
    allFilePath(folderPath,fileList)
    saveFolderPath = os.path.join(saveFile,folder)
    if not os.path.exists(saveFolderPath):
        os.mkdir(saveFolderPath)

    # len1 = len(fileList)
    print(len)
    for i in range(len(fileList)):
        file =fileList[i]
        if not file.endswith(".jpg"):
            continue
        if i%10==0:
            fileName =file.split("\\")[-1]
            fileName1 =str(i)+"_"+fileName
            newFilePath = os.path.join(saveFolderPath,fileName1)
            print(i)
            shutil.move(file,newFilePath)
