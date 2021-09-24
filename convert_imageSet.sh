#!/usr/bin/env sh

/home/xiaolei/caffe/caffe/build/tools/convert_imageset --shuffle=true --encoded=true --encode_type=jpg --resize_height=128 --resize_width=128 "" /home/xiaolei/train_data/myNetTraing/datasets/datasets/pedestrain/gender/train.txt /home/xiaolei/ramdisk/Gender_lmdb
# /home/lxk/caffe-master/build/tools/convert_imageset --shuffle=true --resize_height=128 --resize_width=128 /home/data_ssd/ ./train.txt ./train_lmdb
# /home/lxk/caffe-master/build/tools/convert_imageset --shuffle=true --resize_height=128 --resize_width=128 /home/data_ssd/ ./val.txt ./val_lmdb

# /home/lxk/caffe-master/build/tools/convert_imageset --shuffle=true --gray=true --resize_height=128 --resize_width=128 /home/data_ssd/ ./train.txt ./train_lmdb
# /home/lxk/caffe-master/build/tools/convert_imageset --shuffle=true --gray=true --resize_height=128 --resize_width=128 /home/data_ssd/ ./val.txt ./val_lmdb


# /home/lxk/caffe-master/build/tools/compute_image_mean ./train_lmdb ./train_mean_128x128.binaryproto
# /home/lxk/caffe-master/build/tools/compute_image_mean ./val_lmdb ./val_mean_128x128.binaryproto
# /home/lxk/ZHP/solfware/caffe/build/tools/convert_imageset --shuffle=true --encoded=true --encode_type=jpg --resize_height=128 --resize_width=128 /home/lxk/ ./train.txt ./VeID_lmdb
# /home/lxk/caffe-master/build/tools/convert_imageset --shuffle=true --encoded=true --encode_type=jpg /home/data_ssd/ /home/data_ssd/train.txt /home/data_ssd/VeID_lmdb
 
#./build/tools/convert_imageset --shuffle --add_image=false  --encoded=true --encode_type=jpg \
#/home/xiaokai/image/Face/val/  /home/xiaokai/image/Face/val.txt  /home/xiaokai/caffe-master/DB/val_lmdb_Face_DALI

#/home/lxk/caffe-master/build/tools/compute_image_mean ./ReID_lmdb ./ReID_head_mean_96x96.binaryproto