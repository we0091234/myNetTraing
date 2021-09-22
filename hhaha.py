# def add(a):
#     return a+b


# if __name__=="__main__":
#     b=3
#     print(add(2)
import os
path = r"/home/xiaolei/train_data/myNetTraing/utils"
for root,d,file in os.walk(path):
    print(root,d,file)