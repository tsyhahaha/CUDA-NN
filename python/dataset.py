import os

import torch
from torch.utils.data import Dataset, DataLoader

import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F

import torch.nn.parallel
from torch.autograd import Variable
import numpy as np
import h5py
from tqdm import tqdm

# provider
def shift_point_cloud(batch_data, shift_range=0.1):
    """ Randomly shift point cloud. Shift is per point cloud.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, shifted batch of point clouds
    """
    B, N, C = batch_data.shape
    shifts = np.random.uniform(-shift_range, shift_range, (B,3))
    for batch_index in range(B):
        batch_data[batch_index,:,:] += shifts[batch_index,:]
    return batch_data


def random_scale_point_cloud(batch_data, scale_low=0.8, scale_high=1.25):
    """ Randomly scale the point cloud. Scale is per point cloud.
        Input:
            BxNx3 array, original batch of point clouds
        Return:
            BxNx3 array, scaled batch of point clouds
    """
    B, N, C = batch_data.shape
    scales = np.random.uniform(scale_low, scale_high, B)
    for batch_index in range(B):
        batch_data[batch_index,:,:] *= scales[batch_index]
    return batch_data

def random_point_dropout(batch_pc, max_dropout_ratio=0.875):
    ''' batch_pc: BxNx3 '''
    for b in range(batch_pc.shape[0]):
        dropout_ratio =  np.random.random()*max_dropout_ratio # 0~0.875
        drop_idx = np.where(np.random.random((batch_pc.shape[1]))<=dropout_ratio)[0]
        if len(drop_idx)>0:
            batch_pc[b,drop_idx,:] = batch_pc[b,0,:] # set to the first point
    return batch_pc

class PointCloudDataset(Dataset):
    def __init__(self,root, split):
        self.list_of_points = []
        self.list_of_labels = []
        self.root = root
        self.split = split

        # with h5py.File(f"{split}_point_clouds.h5","r") as hf:
        with h5py.File(f"{self.root}/{self.split}_point_clouds.h5","r") as hf:
            for k in hf.keys():
                self.list_of_points.append(hf[k]["points"][:].astype(np.float32))
                self.list_of_labels.append(hf[k].attrs["label"])
        self.fix_length_statistics_with_median()

    def __len__(self):
        return len(self.list_of_points)

    def __getitem__(self, idx):
        points = self.list_of_points[idx]
        label = self.list_of_labels[idx]
        return points, label

    def fix_length_statistics_with_median(self):
        lengths = [points.shape[0] for points in self.list_of_points]
        fix_length = int( np.median(lengths) )
        
        new_list_of_points = []
        for points in self.list_of_points:
            if(points.shape[0] >= fix_length):
                new_list_of_points.append(points[:fix_length, :])
            else:
                new_list_of_points.append(np.concatenate((points, np.zeros((fix_length - points.shape[0], 3), dtype=np.float32)), axis=0))
        self.list_of_points = new_list_of_points


class PointCloudTestDataset(Dataset):
    def __init__(self, list_of_points, list_of_labels):
        self.list_of_points = []
        self.list_of_labels = []

        self.list_of_labels = list_of_labels
        self.list_of_points = list_of_points

        self.fix_length_statistics_with_median()

    def __len__(self):
        return len(self.list_of_points)

    def __getitem__(self, idx):
        points = self.list_of_points[idx]
        label = self.list_of_labels[idx]
        return points, label

    def fix_length_statistics_with_median(self):
        lengths = [points.shape[0] for points in self.list_of_points]
        fix_length = int(np.median(lengths))
        
        new_list_of_points = []
        for points in self.list_of_points:
            if(points.shape[0] >= fix_length):
                new_list_of_points.append(points[:fix_length, :])
            else:
                new_list_of_points.append(np.concatenate((points, np.zeros((fix_length - points.shape[0], 3), dtype=np.float32)), axis=0))
        self.list_of_points = new_list_of_points

def save_model_params_and_buffers_to_txt(model, directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    for name, param in model.named_parameters():
        np.savetxt(os.path.join(directory, f'{name}.txt'), param.detach().cpu().numpy().flatten())
    
    for name, buffer in model.named_buffers():
        np.savetxt(os.path.join(directory, f'{name}.txt'), buffer.detach().cpu().numpy().flatten())