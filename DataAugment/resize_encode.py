# -*- coding: utf-8 -*
import os
import cv2
import numpy as np
path1=r'/home/data_ssd/yu/VehicleReID-example'
path2=r'/home/data_ssd/ZHP/ReIDData/ReID_Aug'

def resize_1(path1,path2):
	num=0
	for roof, d,filelist in os.walk(path1):
		if not filelist:
			continue
		new=os.path.join(path2,os.path.basename(roof))
		if not os.path.isdir(new):
			os.mkdir(new)
		print(num,roof)
		for pic in filelist:
			pic_path=os.path.join(roof,pic)
			save_path=os.path.join(new,pic[:-4]+'.jpg')
			#img=Image.open(pic_path)
			#src=img.resize((128,128),Image.ANTIALIAS)
			#src.save(save_path)
			##############
			img=cv2.imread(pic_path)
			src=cv2.resize(img,(96,144))
			cv2.imwrite(save_path,src)
		num+=1
			########
			# img=cv2.imdecode(np.fromfile(pic_path,dtype=np.uint8),-1)
			# # src=cv2.resize(img,(52,52),interpolation=cv2.INTER_LINEAR)
			# cv2.imencode('.jpg', img)[1].tofile(save_path)

def resize_2(path1):
	num=0
	for roof, d,filelist in os.walk(path1):
		if not filelist:
			continue
		print(num,roof)
		for pic in filelist:
			pic_path=os.path.join(roof,pic)
			save_path=os.path.join(roof,pic)
			#img=Image.open(pic_path)
			#src=img.resize((128,128),Image.ANTIALIAS)
			#src.save(save_path)
			##############
			img=cv2.imread(pic_path)
			src=cv2.resize(img,(128,128))
			cv2.imwrite(save_path,src)
		num+=1
			########
			# img=cv2.imdecode(np.fromfile(pic_path,dtype=np.uint8),-1)
			# # src=cv2.resize(img,(52,52),interpolation=cv2.INTER_LINEAR)
			# cv2.imencode('.jpg', img)[1].tofile(save_path)
resize_2(path1)

