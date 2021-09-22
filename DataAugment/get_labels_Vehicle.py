# -*- coding: utf-8 -*
import os
import random
import sys
path = r"/home/lxk/ZHP/data/Vehicle/VehicleData_Head"
txtname =os.path.basename(path)+'.txt'
f1=open('Vehicle.txt','w',encoding='utf-8')
f2=open(txtname,'w',encoding='utf-8')
label=0
dirs1 = os.listdir(path)
dirs1.sort()
for filedir in dirs1:
	dirpath = os.path.join(path,filedir)
	dirs = os.listdir(dirpath)
	dirs.sort()
	for filensme in dirs:
		data_path = os.path.join(dirpath,filensme)
		print(label,data_path)
		f1.write(filensme+'\n')
		for img in os.listdir(data_path):
			imgpath = os.path.join(data_path,img)
			labels =imgpath+' '+str(label)+'\n'
			f2.write(labels)
		label+=1
f1.close()
f2.close()