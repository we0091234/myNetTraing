import os
class Category():
    def __init__(self):
           self.sum = 0
           self.right=0
           self.error=0

hehe=[]
for i in range(3):
    haha = Category()
    haha.sum=i
    hehe.append(haha)

for temp in hehe:
    print(temp.sum)