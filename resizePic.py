import cv2
import os

def allFilePath(rootPath,allFIleList):
    fileList = os.listdir(rootPath)
    for temp in fileList:
        if os.path.isfile(os.path.join(rootPath,temp)):
            if temp.endswith(".jpg"):
                allFIleList.append(os.path.join(rootPath,temp))
        else:
            allFilePath(os.path.join(rootPath,temp),allFIleList)


rootPath = r"H:\@pedestrain_datasets\@_pedestrain1\pangshou\2"
savePath = r"C:\train_data\bodySize\2"
picList =[]

allFilePath(rootPath,picList)
i = 0
for temp in picList:
    i+=1
    img = cv2.imread(temp)
    img = cv2.resize(img,(144,144))
    print(i,temp)
    folderName = temp.split("\\")[-2]
    folderpath = os.path.join(savePath,folderName)
    if not os.path.exists(folderpath):
        os.mkdir(folderpath)
    picName = temp.split("\\")[-1]
    newPicPath = os.path.join(folderpath,picName)
    cv2.imwrite(newPicPath,img)

