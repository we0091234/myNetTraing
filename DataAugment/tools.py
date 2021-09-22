import os
import sys
import shutil
import random
from PIL import Image
def num_to_sting(N,ID):
    aa = str(ID)
    num = len(aa)
    oo = '0'
    str_zo = ''
    if (num <= N):
        for nn in range(N - num):
            str_zo = oo + str_zo
        str_name = str_zo + str(ID)
        # print(str_name)
    return str_name

#根据ID聚类移动文件夹
def move_same_ID_NO1(path,save_path,th):
	if not os.path.isdir(save_path):
		os.mkdir(save_path)
	txtpath='rm_repeatID_same_id_'+str(th)+'.txt'
	f=open(txtpath,'r')
	for line in f.readlines():
		lines=line.strip()
		ids=lines.split(' ')
		if len(ids)>1:
			newpath=os.path.join(save_path,ids[0])
			print(newpath)
			if not os.path.isdir(newpath):
				os.mkdir(newpath)
			for id in ids:
				data_path=os.path.join(path,id)
				new_path=os.path.join(newpath,id)
				#print(data_path)
				#shutil.copytree(data_path,new_path)
				shutil.move(data_path,new_path)

#提取分辨率最大的图片保存
def save_max_PPI_NO2(path,save_path):
	if not os.path.isdir(save_path):
		os.mkdir(save_path)
	for paths,d,filelist in os.walk(path):
		if not filelist:
			continue
		dirt1={}
		for filename in filelist:
			data1_path=os.path.join(paths,filename)
			print (data1_path)
			img=Image.open(data1_path)
			imgsize=img.size
			PPI=max(imgsize)*min(imgsize)
			dirt1[data1_path]=PPI
		Max_value=max(dirt1, key=dirt1.get)
		dst_path=save_path+'//'+filename
		shutil.copyfile(Max_value,dst_path)
#随机抽取图片保存
def save_random_img_NO3(path,save_path):
	if not os.path.isdir(save_path):
		os.mkdir(save_path)
	for paths,d,filelist in os.walk(path):
		if not filelist:
			continue
		sample=random.sample(filelist,1)
		for filename in sample:
			data1_path=os.path.join(paths,filename)
			print(data1_path)
			dst_path=save_path+'//'+filename
			shutil.copyfile(data1_path,dst_path)
#删出width比较小的图片
def remove_width_min_NO4(path,width):
	n=0
	for roof,d,filelist in os.walk(path):
		if not filelist:
			continue
		for filename in filelist:
			data1_path=os.path.join(roof,filename)
			print (n,data1_path)
			img=Image.open(data1_path)
			imgsize=img.size
			w=img.size[0]
			h=img.size[1]
			img.close()
			if w<=width or h/w >5 or w > h:
				os.remove(data1_path)
		if len(os.listdir(roof))<4:
			shutil.rmtree(roof)
		n+=1
#rename
def rename_file_NO5(path,value):
	filelist=os.listdir(path)
	num=0
	if value:
		for name in filelist:
			data_path=os.path.join(path,name)
			print(num,data_path)
			newname=os.path.join(path,os.path.basename(path)+num_to_sting(5,str(num)))
			num+=1
			os.rename(data_path,newname)
	else:
		for name in filelist:
			data_path=os.path.join(path,name)
			print(num,data_path)
			filelists=os.listdir(data_path)
			for file in filelists:
				data1_path=os.path.join(data_path,file)
				newname=os.path.join(data_path,name+num_to_sting(6,str(num)))
				num+=1
				os.rename(data1_path,newname)
				if len(os.listdir(newname))<3:
					shutil.rmtree(newname)
					continue
				shutil.move(newname,os.path.join(path,os.path.basename(newname)))
def move_img_NO6(path,save_path):
	if not os.path.isdir(save_path):
		os.mkdir(save_path)
	num=0
	for roof,d,filelist in os.walk(path):
		if not filelist:
			continue
		print(num,roof)
		if len(filelist)<4:
			shutil.rmtree(roof)
			continue
		name=os.path.basename(save_path)
		new_path=save_path+'/'+name+'A'+num_to_sting(5,num)
		shutil.move(roof,new_path)
		num+=1
def move_same_ID_NO7(path,save_path,th):
	if not os.path.isdir(save_path):
		os.mkdir(save_path)
	txtpath='rm_repeatID_same_id_'+str(th)+'.txt'
	f=open(txtpath,'r')
	NUM=0
	for line in f.readlines():
		lines=line.strip()
		ids=lines.split(' ')
		newpath=os.path.join(save_path,num_to_sting(6,NUM))
		print(NUM)
		if len(ids)>1:
			if not os.path.isdir(newpath):
				os.mkdir(newpath)
			for id in ids:
				data_path=os.path.join(path,id.split('_')[1])
				new_path=os.path.join(newpath,id.split('_')[1])
				shutil.move(data_path,new_path)
			NUM+=1
def rename_file_NO8(path,N): 
	flag=0
	img_num=0
	for roof, d, filelist in os.walk(path):
		if not d and not filelist:
			shutil.rmtree(roof)
		if d and filelist:
			os.system("pause")
		if not d and filelist:
			print(flag, roof)
			img_num+=len(filelist)
			num = 0
			for filename in filelist:
				img_path=os.path.join(roof,filename)
				newname=os.path.basename(roof)+'_'+num_to_sting(N,num)+'.jpg'
				num+=1
				new_path=os.path.join(roof,newname)
				os.rename(img_path,new_path)
			flag+=1
	print('image_num: ',img_num)
	print('labels_num: ',flag)

def remove_th_NO9(path,value):
	n=0
	for roof,d,filelist in os.walk(path):
		if not filelist:
			continue
		print (n,roof)
		for img in filelist:
			img_path=os.path.join(roof,img)
			th=img.split('_')[0]
			if float(th)<value:
				os.remove(img_path)
		if len(os.listdir(roof))<6:
			shutil.rmtree(roof)
		n+=1
def save_max_th_NO10(path,save_path):
	if not os.path.isdir(save_path):
		os.mkdir(save_path)
	n=0
	for paths,d,filelist in os.walk(path):
		if not filelist:
			continue
		print(n,paths)
		n+=1
		dirt1={}
		for img in filelist:
			data1_path=os.path.join(paths,img)
			th=img.split('_')[0]
			dirt1[data1_path]=float(th)
		Max_value=max(dirt1, key=dirt1.get)
		dst_path=save_path+'//'+img
		shutil.copyfile(Max_value,dst_path)
def save_max_num_NO11(path,save_path):
	if not os.path.isdir(save_path):
		os.mkdir(save_path)
	n=0
	filelist1=os.listdir(path)
	for list1 in filelist1:
		data1=os.path.join(path,list1)
		print(n,data1)
		filelist2=os.listdir(data1)
		dirt1={}
		for list2 in filelist2:
			data2=os.path.join(data1,list2)
			dirt1[data2]=len(os.listdir(data2))
		Max_value=max(dirt1, key=dirt1.get)
		dst_path=save_path+'//'+os.path.basename(Max_value)
		shutil.move(Max_value,dst_path)
		n+=1
def move_same_ID_NO12(path,save_path,th):
	if not os.path.isdir(save_path):
		os.mkdir(save_path)
	txtpath='rm_repeatID_same_id_'+str(th)+'.txt'
	f=open(txtpath,'r')
	NUM=0
	for line in f.readlines():
		lines=line.strip()
		ids=lines.split(' ')
		newpath=os.path.join(save_path,os.path.basename(path)+'A'+num_to_sting(5,NUM))
		if len(ids)>1:
			if not os.path.isdir(newpath):
				os.mkdir(newpath)
			for id in ids:
				data_path=os.path.join(path,id.split('_')[1])
				print(NUM,data_path)
				for img in os.listdir(data_path):
					imgpath=os.path.join(data_path,img)
					new_path=os.path.join(newpath,img)
					shutil.move(imgpath,new_path)
				if len(os.listdir(data_path))==0:
					shutil.rmtree(data_path)
			NUM+=1
def random_remove_200_NO13(path):
	n=0
	for roof,d,filelist in os.walk(path):
		if not filelist:
			continue
		print (n,roof)
		if len(filelist)>100:
			sample=random.sample(filelist,len(filelist)-100)
			for img in sample:
				img_path=os.path.join(roof,img)
				os.remove(img_path)
		if len(os.listdir(roof))<6:
			shutil.rmtree(roof)
		n+=1
def rm_watermark_NO15(path):
	n=0
	for roof,d,filelist in os.walk(path):
		if not filelist:
			continue
		print(n,roof)
		for img in filelist:
			data1_path=os.path.join(roof,img)
			th=img.split('_')[0]
			if 'ERROR' in img and float(th)>0.999:
				# new=os.path.join(save_path,os.path.basename(roof))
				# if not os.path.isdir(new):
				# 	os.mkdir(new)
				# shutil.move(data1_path,os.path.join(new,img))
				os.remove(data1_path)
		if len(os.listdir(roof))<5:
			shutil.rmtree(roof)
		n+=1

def remove_same_ID_NO16(path,th):
	txtpath='rm_repeatID_same_id_'+str(th)+'.txt'
	f=open(txtpath,'r')
	NUM=0
	for line in f.readlines():
		lines=line.strip()
		ids=lines.split(' ')
		if len(ids)>1:
			dicts={}
			for id in ids:
				data_path=os.path.join(path,id.split('_')[1])
				print(NUM,data_path)
				dicts[data_path]=len(os.listdir(data_path))
			Max_value=max(dicts, key=dicts.get)
			for key,value in dicts.items():
				if key != Max_value:
					shutil.rmtree(key)
			NUM+=1

import zipfile
import time

def zipfile_NO17(startdir):
	#startdir  #要压缩的文件夹路径
	k=0
	file_news = startdir +'.zip' # 压缩后文件夹的名字
	z = zipfile.ZipFile(file_news,'w',zipfile.ZIP_DEFLATED) #参数一：文件夹名
	for dirpath, dirnames, filenames in os.walk(startdir):
	    fpath = dirpath.replace(startdir,'') #这一句很重要，不replace的话，就从根目录开始复制
	    fpath = fpath and fpath + os.sep or ''#这句话理解我也点郁闷，实现当前文件夹以及包含的所有文件的压缩
	    for filename in filenames:
	        z.write(os.path.join(dirpath, filename),fpath+filename)
	    print (k,dirpath)
	    k+=1
	z.close()


#############################################################################################################
falg='NO8'
strs=['move_same_ID_NO1','save_max_PPI_NO2','save_random_img_NO3','remove_width_min_NO4','rename_file_NO5',
      'move_img_NO6','move_same_ID_NO7','rename_file_NO8','remove_th_NO9','save_max_th_NO10',
      'save_max_num_NO11','move_same_ID_NO12','random_remove_200_NO13','rm_watermark_NO15','remove_same_ID_NO16','zipfile_NO17']
if __name__ == '__main__':
	path = r'/home/lxk/ZHP/data/Vehicle/VehicleData2/hz_data'
	save_path= r'./0001'
	for fun in strs:
		if falg in fun:
			eval(fun)(path,6)
			good
