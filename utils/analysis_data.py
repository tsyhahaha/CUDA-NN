import h5py
import numpy as np
import torch

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

# random.shuffle(data)

bz = 2

for i in range(1000//bz):
    batch = []
    labels = []
    for j in range(bz):
        if i*bz + j >= 1000:
            break
        print("point: ", i*bz + j, "labels: ", data[i*bz + j][1])
        labels.append(data[i*bz + j][1])
        point = data[i*bz + j][0].reshape(1, -1, 3)
        point = torch.from_numpy(point)
        point = point[:,:30000,:].transpose(-2, -1)

        pad_length = 30000 - point.size(-1)
        if pad_length > 0:
            point = torch.nn.functional.pad(point, (0, pad_length, 0, 0), mode='constant', value=0)

        batch.append(point)

    batch = torch.cat(batch, dim=0)
    batch = batch.numpy()
    labels = torch.tensor(labels)
    np.savetxt(f"/home/tsyhahaha/CUDA-NN/data/beat/{i}.txt", batch.flatten())
    np.savetxt(f"/home/tsyhahaha/CUDA-NN/data/beat/{i}.shape.txt", np.array(batch.shape, dtype=int), fmt='%d')
    np.savetxt(f"/home/tsyhahaha/CUDA-NN/data/beat/{i}.label.txt", np.array(labels, dtype=int).flatten(), fmt='%d')
