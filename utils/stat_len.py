import h5py
import numpy as np
import torch
from matplotlib import pyplot as plt

import random

def read_h5_file(file_path):
    try:
        dataset = []
        labels=[]
        with h5py.File(file_path, 'r') as file:
            keys = list(file.keys())
            for name in keys:
                dataset.append(np.array(file[name]['points']))
                labels.append(file[name].attrs['label'])
            return dataset, labels
    except Exception as e:
        print(f"Error: {e}")

def get_info(dataset):
    mean_len = 0
    for d in dataset:
        mean_len += d.size // 3
    print(mean_len//len(dataset))

file_path = '/home/tsyhahaha/CUDA-NN/data/splits/test_point_clouds.h5'

data_points, labels = read_h5_file(file_path)

data = [(data_points[i], labels[i]) for i in range(len(labels))]

lst = []
for p, l in data:
    lst.append(len(p))

plt.plot([i for i in range(len(lst))], lst)
plt.savefig('tmp.png')