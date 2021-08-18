import cv2
import os
import numpy as np
def cv_imread(path):
    img=cv2.imdecode(np.fromfile(path,dtype=np.uint8),-1)
    return img

def allFilePath(rootPath,allFIleList):
    fileList = os.listdir(rootPath)
    for temp in fileList:

# foldname=r"D:\trainTemp\Upcolor\train"
# foldname=r"F:\PedestrainAttribute\onePic"
        if os.path.isfile(os.path.join(rootPath,temp)):
            allFIleList.append(os.path.join(rootPath,temp))
        else:
            allFilePath(os.path.join(rootPath,temp),allFIleList)

filePath =r"M:\检测报告\华尊测试1111111111111\吸烟98"
savePath=r"M:\检测报告\华尊测试1111111111111\save"

fileList =[]

allFilePath(filePath,fileList)

for temp in fileList:
    img=cv_imread(temp)
    h,w,c=img.shape
    print(h,w)
    min=0
    max =0
    t=0
    b=0
    l=0
    r=0
    if h>w:
        l=int((h-w)/2)
        r=l
        # img = cv2.copyMakeBorder(img, l,  r, l,  r, cv2.BORDER_CONSTANT, value=(128, 128, 128))
    else:
        b= int((w-h) / 2)
        t = b
        # img = cv2.copyMakeBorder(img, b, t,  b,  t, cv2.BORDER_CONSTANT, value=(128, 128, 128))
    img = cv2.copyMakeBorder(img, int(0.8*b), int(0.8*t), int(2*l), int(2*r), cv2.BORDER_CONSTANT,value=(128,128,128))
    # cv2.namedWindow("haha")
    # cv2.imshow("haha",img)
    # cv2.waitKey(0)
    name=temp.split("\\")[-1]
    save_path=os.path.join(savePath,name)
    cv2.imencode('.jpg', img)[1].tofile(save_path)
