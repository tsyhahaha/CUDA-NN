# 这是程序一的Python模板程序，您可以增加epoch后提交该程序而不进行任何修改（如果不在意准确率的分数），对于（加分题），请参考程序二的模板程序自行实现

'''
Package                  Version
------------------------ ----------
certifi                  2024.8.30
charset-normalizer       3.3.2
cmake                    3.30..3
filelock                 3.16.0
h5py                     3.11.0
hdf5                     1.12.1
idna                     3.8
Jinja2                   3.1.4
lit                      18.1.8
MarkupSafe               2.1.5
mpmath                   1.3.0
networkx                 3.3
numpy                    1.26.0
nvidia-cublas-cu11       11.10.3.66
nvidia-cuda-cupti-cu11   11.7.101
nvidia-cuda-nvrtc-cu11   11.7.99
nvidia-cuda-runtime-cu11 11.7.99
nvidia-cudnn-cu11        8.5.0.96
nvidia-cufft-cu11        10.9.0.58
nvidia-curand-cu11       10.2.10.91
nvidia-cusolver-cu11     11.4.0.1
nvidia-cusparse-cu11     11.7.4.91
nvidia-nccl-cu11         2.14.3
nvidia-nvtx-cu11         11.7.91
Pillow                   10.4.0
pip                      24.2
requests                 2.32.3
setuptools               72.1.0
sympy                    1.13.2
torch                    2.0.1
torchaudio               2.0.2
torchvision              0.15.2
triton                   2.0.0
typing_extensions        4.12.2
urllib3                  2.2.2
wheel                    0.43.0
'''

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

from dataset import PointCloudDataset, random_point_dropout, random_scale_point_cloud, shift_point_cloud, save_model_params_and_buffers_to_txt
from model import PointNet, PointNetLoss
from inference import load_model_params_and_buffers_from_txt

# import provider
num_class = 10
total_epoch = 30
script_dir = os.path.dirname(__file__)  # 获取脚本所在的目录

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True


# def test(model, loader, num_class=10):
#     mean_correct = []
#     classifier = model.eval()

#     # for j, (points, target) in tqdm(enumerate(loader), total=len(loader)): #显示进度条
#     for j, (points, target) in enumerate(loader):

#         points, target = points.cuda(), target.cuda()

#         points = points.transpose(2, 1)
#         pred, _ = classifier(points)
#         pred_choice = pred.data.max(1)[1]

#         correct = pred_choice.eq(target.long().data).cpu().sum()
#         mean_correct.append(correct.item())

#     instance_acc = np.sum(mean_correct) / len(loader.dataset)

#     return instance_acc

# def pad_collate_fn(batch):
#     # 找到批次中最小的数组大小
#     min_size = min([item[0].shape[0] for item in batch])
    
#     # 截断数组
#     padded_batch = []
#     for points, target in batch:
#         # 截断数组
#         points = points[:min_size, :]
#         padded_batch.append((points, target))
    
#     # 使用默认的 collate_fn 处理填充后的批次
#     return torch.utils.data.dataloader.default_collate(padded_batch)



def main():
    # 创建数据集实例
    data_path = '../data/splits'

    train_dataset = PointCloudDataset(root=data_path, split='train')
    # test_dataset = PointCloudDataset(root=data_path, split='test')

    # 创建 DataLoader 实例
    # train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=10, drop_last=True, collate_fn=pad_collate_fn) #batch_size内固定长度截取
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=10, drop_last=True) #全局固定长度填充/截取
    # test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=10, drop_last=False)

    print("finish DATA LOADING")

    '''MODEL LOADING'''

    classifier = PointNet(num_class)
    criterion = PointNetLoss()
    classifier.apply(inplace_relu)

    load_model_params_and_buffers_from_txt(classifier, '/home/tsyhahaha/default')

    classifier = classifier.cuda()
    criterion = criterion.cuda()

    optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

    global_step = 0
    best_instance_acc = 0.0
    best_class_acc = 0.0
    print("finish MODEL LOADING")

    '''TRANING'''
    print("start TRANING")

    for epoch in range(total_epoch):
        print('Epoch %d (%d/%s):' % (epoch + 1, epoch + 1, total_epoch))
        mean_correct = []
        classifier = classifier.eval()

        # for batch_id, (points, target) in tqdm(enumerate(train_dataloader, 0), total=len(train_dataloader), smoothing=0.9): #显示进度条
        for batch_id, (points, target) in enumerate(train_dataloader, 0):
            optimizer.zero_grad()

            points = points.data.numpy()
            points = random_point_dropout(points)
            points[:, :, 0:3] = random_scale_point_cloud(points[:, :, 0:3])
            points[:, :, 0:3] = shift_point_cloud(points[:, :, 0:3])
            points = torch.Tensor(points)
            points = points.transpose(2, 1)

            points, target = points.cuda(), target.cuda()

            pred, trans_feat = classifier(points)
            print(pred)
            loss, base_loss, mat_diff_loss = criterion(pred, target.long(), trans_feat)
            print(f"total loss: {loss}, base_loss: {base_loss}, mat_diff_loss: {mat_diff_loss}")
            pred_choice = pred.data.max(1)[1]

            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(points.size()[0]))
            loss.backward()
            optimizer.step()
            global_step += 1
            import pdb; pdb.set_trace()

        scheduler.step()

        train_instance_acc = np.mean(mean_correct)

        print('Train Instance Accuracy: %f' % train_instance_acc)

    print("finish TRANING")
    save_model_params_and_buffers_to_txt(classifier, script_dir)

if __name__ == '__main__':
    main()