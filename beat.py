import os
import time
import yaml

import logging
import random
import argparse

import subprocess

import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.parallel
import numpy as np

from python.model import PointNet

beat_dir = ""

def gen(cfg):
    num, save_dir = cfg['data_num'], beat_dir
    print("-"*50)
    print(f"Generate test points for {cfg['layer'].upper()}")
    print("-"*50)
    for i in range(num):
        shape = ()
        B, L = random.randint(1, 16), random.randint(32, 128)
        if cfg['layer'] == "linear":
            channel = cfg['linear']['in_features']
            shape = (B, channel)
        elif cfg['layer'] in ["conv1d", "batchnorm1d"]:
            channel = cfg[cfg['layer']]['in_channels']
            shape = (B, channel, L)
        else:
            raise ValueError(f"cfg[{cfg['layer']}] not implemented!")
        data_point = np.random.rand(*shape)
        np.savetxt(os.path.join(save_dir, f'{i+1}.txt'), data_point.flatten())
        np.savetxt(os.path.join(save_dir, f'{i+1}.shape.txt'), np.array(shape).flatten(), fmt="%d")

def test_py(cfg):
    pyout_dir = cfg['data_dir'] + "/pyout"
    os.makedirs(pyout_dir, exist_ok=True)

    if cfg['layer'] == 'linear':
        net = nn.Linear(cfg['linear']['in_features'], cfg['linear']['out_features'])
    elif cfg['layer'] == 'conv1d':
        net = nn.Conv1d(cfg['conv1d']['in_channels'], cfg['conv1d']['out_channels'], 1)
    elif cfg['layer'] == 'batchnorm1d':
        net = nn.BatchNorm1d(cfg['batchnorm1d']['in_channels'])
    else:
        raise ValueError(f"layer {cfg['layer']} not implemented!")

    net.eval().cuda().to(torch.float32)
    param_path = cfg['param_path']
    weights = np.loadtxt(param_path + ".weight.txt", dtype=np.float32)
    bias = np.loadtxt(param_path + ".bias.txt", dtype=np.float32)
    
    net.weight.data = torch.from_numpy(weights).cuda().reshape(net.weight.shape)
    net.bias.data = torch.from_numpy(bias).cuda().reshape(net.bias.data.shape)

    if cfg['layer'] == 'batchnorm1d':
        running_mean = np.loadtxt(param_path + ".running_mean.txt", dtype=np.float32)
        running_var = np.loadtxt(param_path + ".running_var.txt", dtype=np.float32)
        net.running_mean = torch.from_numpy(running_mean).cuda()
        net.running_var = torch.from_numpy(running_var).cuda()


    print(f"Test(python) module {cfg['layer'].upper()}")
    print("-"*50)
    
    files = os.listdir(beat_dir)
    files = [s[:-4] for s in files if 'shape' not in s]
    for file in files:
        out_file = os.path.join(pyout_dir, file+".txt")
        file = os.path.join(beat_dir, file)
        shape = tuple(np.loadtxt(file+".shape.txt", dtype=int))
        data = np.loadtxt(file+".txt", dtype=np.float32)
        data = torch.from_numpy(data).reshape(shape).cuda()
        output = net(data.to(torch.float32))
        np.savetxt(out_file, output.detach().cpu().numpy().flatten(), fmt='%.06f')


def test_cu(cfg):
    cuout_dir = cfg['data_dir'] + "/cuout"
    os.makedirs(cuout_dir, exist_ok=True)
    print(f"Test(cuda) module {cfg['layer'].upper()}")
    print("-"*50)

    cur = os.getcwd()
    target = os.path.join(cur, 'CUDA-NN/build')
    os.chdir(target)
    
    subprocess.run(['make', 'run'], capture_output=True, text=True)
    os.chdir(cur)

def beat(pyout, cuout):
    print(f"Beat(cuda) the outputs")
    print("-"*50)
    def _check_file(f1, f2):
        data1 = np.loadtxt(f1, dtype=float)
        data2 = np.loadtxt(f2, dtype=float)
        return (np.abs(data1 - data2) < 1e-2).all()
    files = os.listdir(pyout)
    error_list = []
    for file in files:
        f1 = os.path.join(pyout, file)
        f2 = os.path.join(cuout, file)
        if not _check_file(f1, f2):
            error_list.append(file)
    error_list = sorted(error_list, key=lambda x: int(x.split('.')[0]))
    return error_list

def main():
    with open('/home/course/taosiyuan241/CUDA-NN/config.yaml','r') as f:
        cfg = yaml.safe_load(f)
    global beat_dir
    beat_dir = cfg['data_dir'] + "/beat"
    os.makedirs(beat_dir, exist_ok=True)

    if cfg['regenerate']:
        gen(cfg)

    test_py(cfg)
    test_cu(cfg)

    el = beat(cfg['data_dir'] + "/pyout", cfg['data_dir'] + "/cuout")    
    if not el:
        print(f"[AC] Test all {cfg['data_num']} points successfully!")
    else:
        print("please check test point as follow:")
        for ef in el:
            print(ef)
        with open("./error_points.txt", 'w') as f:
            f.write('\n'.join(el))

if __name__=='__main__':
    main()