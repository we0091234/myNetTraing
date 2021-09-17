import os
import shutil

def allFilePath(rootPath,allFIleList):
    fileList = os.listdir(rootPath)
    for temp in fileList:
        if os.path.isfile(os.path.join(rootPath,temp)):
            allFIleList.append(os.path.join(rootPath,temp))
        else:
            allFilePath(os.path.join(rootPath,temp),allFIleList)

fileFolder = r"/home/xiaolei/ramdisk/gender/train1/1"
# fileFolder = r"/home/xiaolei/train_data/HarzoneData/trianRar/"
saveFolder = r"/home/xiaolei/ramdisk/gender/oneFolder/1"

fileList =[]

allFilePath(fileFolder,fileList)
ic=0
for file in fileList:
    if file.endswith(".jpg"):
        jpgPath = file
        # txtPath = file.replace(".jpg",".txt")
        if os.path.exists(file):
            ic+=1
            picName = file.split("/")[-1]
            newPicName=str(ic)+"_"+picName
            newPicPath = os.path.join(saveFolder,newPicName)
            # newTxtPath = newPicPath.replace(".jpg",".txt")
            # print(file,newPicPath)
            # print(txtPath,newTxtPath)
            print(ic,file)
            shutil.move(file,newPicPath)
            # shutil.copy(txtPath,newTxtPath)


print("{} pic done".format(ic) )
