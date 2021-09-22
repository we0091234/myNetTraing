# -*- coding: utf-8 -*
import os
import random
import sys
path = r"/home/lxk/ZHP/data/ReIDData/ReID_Data/YTData2"
txtname =os.path.basename(path)+'.txt'
f=open('Vehicle.txt','w',encoding='utf-8')
f1=open(txtname,'w',encoding='utf-8')
f2=open('val.txt','w',encoding='utf-8')
g = os.walk(path)
label=0
for roof, d, filelist in g:
    if not filelist:
        continue
    print(label, roof)
    if  d:
        os.system('pause')
    vehicle=os.path.basename(roof)+'\n'
    f.write(vehicle)
    for filename in filelist:
        if '.txt' not in filename:
            train_path=os.path.join(roof,filename)
            # train_dir=train_path.split('/')
            # train_dir.pop(0)
            # train_dir.pop(0)
            # train_dir.pop(0)
            # train_data='/'.join(train_dir)
            # img_path=train_data+' '+str(label)+'\n'
            img_path=train_path+'\n'
            f1.write(img_path)
    # sample=random.sample(filelist,2)
    # for val in sample:
    #     val_path=os.path.join(roof,val)
    #     val_dir=val_path.split('/')
    #     val_dir.pop(0)
    #     val_dir.pop(0)
    #     val_dir.pop(0)
    #     val_data='/'.join(val_dir)
    #     img_path=val_data+' '+str(label)+'\n'
    #     f2.write(img_path)
    label+=1
f.close()
f1.close()
f2.close()