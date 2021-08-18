import os

txt1=r"E:\TensorRT\tensorrtx-master\yolov3\build\new\lr1.txt"
txt2=r"E:\TensorRT\tensorrtx-master\yolov3\build\new\lr3.txt"
txt3=r"E:\TensorRT\tensorrtx-master\yolov3\build\ori\lr1-lr3.txt"

f1=open(txt1,"r")
f2=open(txt2,"r")
f3=open(txt3,"w")
while True:
    line1=f1.readline().strip()
    line2=f2.readline().strip()
    if len(line1)>1:
        intline1=float(line1)
        intline2 = float(line2)
    if not line1:
        break
    line3=intline1-intline2
    # line3=round(line3,4)
    strline3=format(line3,".4f")
    # strline3=str(line3)
    f3.write(strline3+"\n")
