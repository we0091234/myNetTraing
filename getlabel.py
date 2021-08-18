import os
import shutil

def allFilePath(rootPath,allFIleList):
    fileList = os.listdir(rootPath)
    for temp in fileList:
        if os.path.isfile(os.path.join(rootPath,temp)):
            allFIleList.append(os.path.join(rootPath,temp))
        else:
            allFilePath(os.path.join(rootPath,temp),allFIleList)


rootPath=r"F:\Driver\train"
folder=os.listdir(rootPath)
i =0
f= open(r"C:\train_data\isDriver\label.txt","w")
for temp in folder:
    folderPath  =os.path.join(rootPath,temp)
    fileList=[]
    allFilePath(folderPath,fileList)
    for file in fileList:
        pathLabel=file+" "+str(i)
        f.write(pathLabel+"\n")
    i=i+1
f.close()

# f=open(r"C:\train_data\isDriver\label.txt","r")
#
# while True:
#     line=f.readline().strip("\n")
#     if not line:
#         break
#     # print(line.strip("\n"))
#     imgpath=line.split(" ")[0]
#     label =line.split(" ")[1]
#     print(label)


