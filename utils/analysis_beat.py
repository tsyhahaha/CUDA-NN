import numpy as np


num=1

f1 = f"/home/tsyhahaha/CUDA-NN/data/cuout/{num}.txt"
f2 = f"/home/tsyhahaha/CUDA-NN/data/pyout/{num}.txt"

a = np.loadtxt(f1)
b = np.loadtxt(f2)

thres = 0.002


num = 0
for i in range(len(a)):
    if abs(a[i]-b[i]) > thres:
        print("(", i//(1024*124), i%(1024*6124)//1024, i%124, ")", end="")
        num+=1

print(np.max(np.abs(a-b)))
print(num)
