import os
import os.path

def allFilePath(rootPath,allFIleList):
    fileList = os.listdir(rootPath)
    for temp in fileList:
        if os.path.isfile(os.path.join(rootPath,temp)):
            allFIleList.append(os.path.join(rootPath,temp))
        else:
            allFilePath(os.path.join(rootPath,temp),allFIleList)

root = "/home/xiaolei/train_data/myNetTraing/datasets/datasets/pedestrain/gender/"
trainOrVal=["train","val"]
total=0
for t in  trainOrVal:
    if not os.path.exists(root+t):
        continue
    folderlist=os.listdir(root+t)
    folderlist.sort()
    num = 0
    txtpathname = root+t+".txt"
    f = open(txtpathname,"w")
    for temp in folderlist:
        kidpathname=os.path.join(root+t,temp)
        kidfilelist=[]
        allFilePath(kidpathname,kidfilelist)
        # kidfilelist = os.listdir(kidpathname)
        for temp1 in kidfilelist:
            string =temp1+" "+str(num)+"\n"
            total=total+1
            print ("%d %s"%(total,string),end="")
            f.write(string)
        num=num+1
    f.close()