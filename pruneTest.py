import torch
import numpy as np
import random

a = np.arange(10)
print(a)
random.shuffle(a)
print(a)
b =torch.from_numpy(a)
# a = torch.randn([15]).abs()
# print(a)
mask = b.gt(3).float()
print(mask)
# mask2=1-mask
# # mask2 = 1-mask
# print(mask)
# print(int(sum(mask))%2==0)
# print(mask2)
# b=a*mask2
# print(b)
# m,n = torch.sort(b)
# print(n)
# id = n[-2:]
# print(id)
# mask[id.tolist()]=1
# print(mask)

remain = int(torch.sum(mask)) % 4
remain2 = 4 - remain
if remain != 0:
    mask1 = 1 - mask
    weightRemain = a * mask1
    b,index = torch.sort(weightRemain)
    idx = index[-remain:]
    mask[idx.tolist()]=1
print(mask)