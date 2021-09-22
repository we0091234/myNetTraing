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
        print(str_name)
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
	for paths,d,filelist in os.walk(path):
		if not filelist:
			continue
		for filename in filelist:
			data1_path=os.path.join(paths,filename)
			print (data1_path)
			img=Image.open(data1_path)
			imgsize=img.size
			PPI=img.size[0]
			img.close()
			if PPI<=width:
				os.remove(data1_path)
#rename
def rename_file_NO5(path):
	filelist=os.listdir(path)
	num=0
	for name in filelist:
		data_path=os.path.join(path,name)
		print(num,data_path)
	# newname=os.path.join(path,'WILD_'+num_to_sting(4,str(num)))
	# names=name.replace('__','_')
		newname=os.path.join(path,'V'+num_to_sting(6,num))
		num+=1
		os.rename(data_path,newname)
def remove_NO6(path):
	for paths,d,filelist in os.walk(path):
		if not filelist:
			continue
		print(paths)
		if len(filelist)<4:
			shutil.rmtree(paths)
falg='NO5'
strs=['move_same_ID_NO1','save_max_PPI_NO2','save_random_img_NO3','remove_width_min_NO4','rename_file_NO5','remove_NO6']
if __name__ == '__main__':
	path = r'/home/lxk/ZHP/data/VeIDWindowData/VeIDWinData/V2_Window'
	save_path= r'../max_PPI'
	for fun in strs:
		if falg in fun:
			eval(fun)(path)


