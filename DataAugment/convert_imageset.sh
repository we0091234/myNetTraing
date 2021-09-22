#!/usr/bin/env sh

/home/lxk/caffe-master/build/tools/convert_imageset --shuffle=true --encoded=true --encode_type=jpg --resize_height=144 --resize_width=96 /home/data_ssd/ ./train.txt ./ReID_lmdb
# /home/lxk/ZHP/solfware/caffe/build/tools/convert_imageset --shuffle=true --encoded=true --encode_type=jpg --resize_height=128 --resize_width=128 /home/lxk/ ./train.txt ./VeID_lmdb
# /home/lxk/caffe-master/build/tools/convert_imageset --shuffle=true --encoded=true --encode_type=jpg /home/data_ssd/ /home/data_ssd/train.txt /home/data_ssd/VeID_lmdb
 
#./build/tools/convert_imageset --shuffle --add_image=false  --encoded=true --encode_type=jpg \
#/home/xiaokai/image/Face/val/  /home/xiaokai/image/Face/val.txt  /home/xiaokai/caffe-master/DB/val_lmdb_Face_DALI