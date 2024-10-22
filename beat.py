import os
import re
import time
import yaml

import logging
import random
import argparse
import subprocess
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.parallel
import numpy as np

from python.model import PointNet

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

beat_dir = ""
target = ""
name = ""
data_num = 0

def clear_directory(dir_path):
    for filename in os.listdir(dir_path):
        file_path = os.path.join(dir_path, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)  # 删除文件
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

def gen(cfg):
    clear_directory(beat_dir)
    num, save_dir = cfg['data_num'], beat_dir
    print(f"Generate test points for {cfg['name'].upper()}")
    for i in tqdm(range(num)):
        shape = ()
        B, L = random.randint(1, 8), random.randint(128, 256)
        if cfg['target'] == 'model':
            if cfg['name'] in ["pointnet", 'stn3d', 'encoder']:
                shape = (B, 3, L)
            elif cfg['name'] == 'stnkd':
                shape = (B, 64, L)
            else:
                raise ValueError(f"cfg[{cfg['name']}] not implemented!")
        elif cfg['target'] == 'layer':
            if cfg['name'] == "linear":
                channel = cfg['linear']['in_features']
                shape = (B, channel)
            elif cfg['name'] in ["conv1d", "batchnorm1d"]:
                channel = cfg[cfg['name']]['in_channels']
                shape = (B, channel, L)
            else:
                raise ValueError(f"cfg[{cfg['name']}] not implemented!")
        elif cfg['target'] == 'op':
            if cfg['name'] == 'max':
                channel = random.randint(3, 64)
                shape = (B, channel, L)
        data_point = np.random.rand(*shape)
        np.savetxt(os.path.join(save_dir, f'{i+1}.txt'), data_point.flatten())
        np.savetxt(os.path.join(save_dir, f'{i+1}.shape.txt'), np.array(shape).flatten(), fmt="%d")
    print("-"*50)
    

def load_model_params_and_buffers_from_txt(model, directory):
    for name, param in model.named_parameters():
        file_path = os.path.join(directory, f'{name}.txt')
        if os.path.exists(file_path):
            param_data = np.loadtxt(file_path)
            param_data = param_data.reshape(param.data.shape)
            param.data = torch.tensor(param_data, dtype=torch.float32)
        else:
            print(f"Warning: {file_path} does not exist, skipping.")

    for name, buffer in model.named_buffers():
        file_path = os.path.join(directory, f'{name}.txt')
        if os.path.exists(file_path):
            buffer_data = np.loadtxt(file_path)
            buffer_data = buffer_data.reshape(buffer.shape)
            buffer.data = torch.tensor(buffer_data, dtype=buffer.dtype)
        else:
            print(f"Warning: {file_path} does not exist, skipping.")
    return model

def test_py(cfg):
    print(f"Test(python) module {cfg['name'].upper()}")
    pyout_dir = cfg['data_dir'] + "/pyout"
    os.makedirs(pyout_dir, exist_ok=True)
    clear_directory(pyout_dir)

    # if test the model
    if cfg['target'] == 'model':
        net = PointNet()
        net = load_model_params_and_buffers_from_txt(net, cfg['param_path'])
        if cfg['name'] == 'stn3d':
            net = net.feat.stn
        elif cfg['name'] == 'stnkd':
            net = net.feat.fstn
        elif cfg['name'] == 'encoder':
            net = net.feat
    elif cfg['target'] == 'op':
        if cfg['name'] == 'max':
            net = lambda x: torch.max(x, 2, keepdim=False)[0]
    elif cfg['target'] == 'layer':
        if cfg['name'] == 'linear':
            net = nn.Linear(cfg['linear']['in_features'], cfg['linear']['out_features'])
        elif cfg['name'] == 'conv1d':
            net = nn.Conv1d(cfg['conv1d']['in_channels'], cfg['conv1d']['out_channels'], 1)
        elif cfg['name'] == 'batchnorm1d':
            net = nn.BatchNorm1d(cfg['batchnorm1d']['in_channels'])
        else:
            raise ValueError(f"layer {cfg['name']} not implemented!")

        param_path = cfg['param_path']
        weights = np.loadtxt(param_path + ".weight.txt", dtype=np.float32)
        bias = np.loadtxt(param_path + ".bias.txt", dtype=np.float32)
        
        net.weight.data = torch.from_numpy(weights).cuda().reshape(net.weight.shape)
        net.bias.data = torch.from_numpy(bias).cuda().reshape(net.bias.data.shape)
        # net.bias.data = torch.zeros(net.bias.data.shape).cuda()

        if cfg['name'] == 'batchnorm1d':
            running_mean = np.loadtxt(param_path + ".running_mean.txt", dtype=np.float32)
            running_var = np.loadtxt(param_path + ".running_var.txt", dtype=np.float32)
            net.running_mean = torch.from_numpy(running_mean).cuda()
            net.running_var = torch.from_numpy(running_var).cuda()
    
    if isinstance(net, nn.Module):
        net = net.eval().cuda().float()

    files = os.listdir(beat_dir)
    files = [s[:-4] for s in files if 'shape' not in s]

    time_sum = 0

    for i in tqdm(range(min(data_num, len(files)))):
        file = files[i]
        out_file = os.path.join(pyout_dir, file+".txt")
        file = os.path.join(beat_dir, file)
        shape = tuple(np.loadtxt(file+".shape.txt", dtype=int))
        data = np.loadtxt(file+".txt", dtype=np.float32)
        data = torch.from_numpy(data).reshape(shape).cuda()
        time_start = time.time()
        if cfg['name'] == 'pointnet':
            output, _ = net(data.to(torch.float32))
            if cfg['pointnet']['argmax']:
                output = torch.argmax(output, dim=1)
        elif cfg['name'] == 'encoder':
            output, _, __ = net(data.to(torch.float32))
        else:
            output = net(data.to(torch.float32))
        time_end = time.time()
        time_sum += time_end - time_start
        np.savetxt(out_file, output.detach().cpu().numpy().flatten(), fmt='%.06f')
    print("Average time cost: ", round(time_sum/len(files), 4),'s')
    print("-"*50)

def test_cu(cfg):
    cuout_dir = cfg['data_dir'] + "/cuout"
    os.makedirs(cuout_dir, exist_ok=True)
    clear_directory(cuout_dir)

    print(f"Test(cuda) module {cfg['name'].upper()}")

    cur = os.getcwd()
    target = os.path.join(cur, 'CUDA-NN/build')
    os.chdir(target)
    
    result = subprocess.run(['make', 'run'], capture_output=True, text=True)
    if result.returncode != 0:
        print("[ERROR] Failed to launch the cuda program!")
        exit(-1)
    
    pattern = r'Average time consumed: ([0-9]*\.[0-9]+) s'
    content = result.stdout
    matches = re.findall(pattern, content)
    for match in matches:
        print(f'Extracted average time: {match}')
    os.chdir(cur)
    print("-"*50)

def beat(pyout, cuout, cfg):
    print(f"Beat the outputs")
    
    if target == 'model' and name == 'pointnet' and cfg['pointnet']['argmax']:
        print(" - using argmax to compare the output of pointnet")

    def _check_file(f1, f2):
        data1 = np.loadtxt(f1, dtype=float)
        data2 = np.loadtxt(f2, dtype=float)
        if data1.size != data2.size:
            raise ValueError(f"{f1} data size not equal: {data1.size}!={data2.size}")
        thres = 1e-3
        if name=='pointnet':
            thres = 1e-1
        mask = np.where(np.abs(data1) > 1e-6, 1, 0)
        l1_error = np.abs(data1 - data2)
        mean_error = np.sum(l1_error * mask) / np.sum(mask)
        is_error = mean_error >= thres
        error_num = np.sum(np.where(l1_error > thres, 1, 0))

        string = "mean_error={%.4f}, error_rate={%f}"%(mean_error, error_num/data1.size)
        return is_error.any(), error_num <= (data1.size//2), string
    
    files = os.listdir(pyout)
    partial_dict = {}
    error_list = []


    for i in tqdm(range(min(data_num, len(files)))):
        file = files[i]
        f1 = os.path.join(pyout, file)
        f2 = os.path.join(cuout, file)
        wrong, partial, output = _check_file(f1, f2)
        if wrong and partial:
            partial_dict.update({file: output})
        elif wrong:
            error_list.append(file)
    error_list = sorted(error_list, key=lambda x: int(x.split('.')[0]))
    partial_items = sorted(partial_dict.items(), key=lambda x: int(x[0].split('.')[0]))
    print("-"*50)
    return partial_items, error_list

def main():
    with open('./config.yaml','r') as f:
        cfg = yaml.safe_load(f)
    global beat_dir, target, name, data_num
    data_num = cfg['data_num']
    target = cfg['target']
    name = cfg['name']
    beat_dir = cfg['data_dir'] + "/beat"
    os.makedirs(beat_dir, exist_ok=True)

    print("-"*50)
    if cfg['mode'] == "regenerate":
        gen(cfg)
    if cfg['mode'] in ['regenerate', 'test']:
        test_py(cfg)
        torch.cuda.empty_cache()
        test_cu(cfg)
    
    pl, el = beat(cfg['data_dir'] + "/pyout", cfg['data_dir'] + "/cuout", cfg)    
    if len(el)==0 and len(pl)==0:
        print(f"[AC] Test all {cfg['data_num']} points successfully!")
    else:
        print("please check test point as follow:")
        print('Partial error test-points:')
        for key, value in pl:
            print(key, ": \t", value)
        print('Wrong answer test-points:')
        for ef in el:
            print(ef)
        with open("./error_points.txt", 'w') as f:
            f.write('Partial List:\n')
            f.write('\n'.join([key+":\t"+value for key, value in pl]))
            f.write('Error List:\n')
            f.write('\n'.join(el))

if __name__=='__main__':
    main()
