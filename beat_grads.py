import os
import yaml
import numpy as np

import torch
from einops import rearrange, repeat, einsum

root = '/home/tsyhahaha/CUDA-NN/data/grads'
cfg_path = '/home/tsyhahaha/CUDA-NN/config_pointnet.yaml'
bz = int(os.getenv("BATCH_SIZE", 4))
cs = int(os.getenv("CROPPING_SIZE", 20000))
target = os.getenv("TARGET", 'linear')   # linear, bn, conv

"""load yaml config"""
def load_yaml_file(file_path):
    with open(file_path, 'r') as file:
        yaml_content = yaml.safe_load(file)
    return yaml_content

def parse_yaml_to_dict(yaml_content):
    def recursive_parse(data, prefix=""):
        layers = {}
        for key, value in data.items():
            new_prefix = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                layers.update(recursive_parse(value, new_prefix))
            else:
                layers[prefix] = {k: int(v) for k,v in data.items()}
                break
        return layers
    return recursive_parse(yaml_content)

"""utils funcs"""
def read_matrix_from_txt(file_path, shape):
    try:
        data = np.loadtxt(file_path, dtype=np.float32).reshape(shape)
        return torch.from_numpy(data)
    except Exception as e:
        raise ValueError(f"Error reading matrix from {file_path} {shape}: {e}")
    
def save_tensor_to_txt(tensor, file_path):
    try:
        data = tensor.cpu().numpy().flatten()
        
        np.savetxt(file_path, data, fmt='%0.6f')
        print(f"Tensor saved to {file_path}")
    except Exception as e:
        raise ValueError(f"Error saving tensor to {file_path}: {e}")

def compare_matrices(matrix_a, matrix_b, tolerance=5e-2):
    # ∣input−other∣ ≤ atol+rtol×∣other∣
    return torch.allclose(matrix_a, matrix_b, atol=tolerance)

def compare(d_in, d_weights, d_bias, d_in_py, d_weights_py, d_bias_py):
    result1 = compare_matrices(d_in, d_in_py)
    result2 = compare_matrices(d_weights, d_weights_py)
    result3 = compare_matrices(d_bias, d_bias_py)
    return result1, result2, result3

def compare_bn(d_in, d_weights, d_bias, d_mu, d_var, d_in_py, d_weights_py, d_bias_py, d_mu_py, d_var_py):
    result1 = compare_matrices(d_in, d_in_py)
    result2 = compare_matrices(d_weights, d_weights_py)
    result3 = compare_matrices(d_bias, d_bias_py)
    result4 = compare_matrices(d_mu, d_mu_py)
    result5 = compare_matrices(d_var, d_var_py)
    return result1, result2, result3, result4, result5


"""beat the computation of grads"""

def linear_backprop(d_out, input, weights):
    """
    d_out: (N, C_out)
    input: (N, C_in)
    weights: (C_out, C_in)
    """
    d_weights = d_out.T @ input     # (C_out, C_in)
    d_bias = torch.sum(d_out, dim=0)   # (C_out)
    d_in = d_out @ weights
    return d_in, d_weights, d_bias

def conv1d_backprop(d_out, input, weights):
    """
    d_out: (N, C_out, L)
    input: (N, C_in, L)
    weights: (C_out, C_in)
    """
    bz = d_out.shape[0]
    input_trans = rearrange(input, 'n c l -> n l c')
    d_weights = torch.sum(torch.bmm(d_out, input_trans), dim=0)
    d_bias = torch.sum(d_out, dim=(0, 2))
    n_weights = repeat(weights, 'm n -> b m n', b=bz).transpose(-1, -2)
    d_in = torch.bmm(n_weights, d_out)
    # d_in = einsum(weights, d_out, 'm n, b n l -> b m l')
    return d_in, d_weights, d_bias


def batchnorm1d_backprop(d_out, weights, x_hat, x_minus_mu, sqrt_var_inv):
    """
    d_out: (N, C, L)/(N, C)
    input: (N, C, L)/(N, C)
    weights: (C)
    x_hat: (N, C, L)/(N, C)
    x_minus_mu: (N, C, L)/(N, C)
    sqrt_var_inv: (C)
    """
    bz = d_out.shape[0]
    if len(d_out.shape) == 3:
        dim = (0, 2)
    else:
        dim = (0)
    d_weights = torch.sum(d_out * x_hat, dim=dim)
    d_bias = torch.sum(d_out, dim=dim)

    if len(d_out.shape) == 3:
        d_x_hat = d_out * weights[None, :, None]
        d_mu = torch.sum(- d_x_hat * sqrt_var_inv[None,:,None], dim=dim)
        d_var = torch.sum(- 0.5 * d_x_hat * x_minus_mu * sqrt_var_inv[None,:,None]**3, dim=dim)
        d_in = d_x_hat * sqrt_var_inv[None,:,None] + d_mu[None,:,None] / bz + 2 * d_var[None, :, None] * (x_minus_mu) / bz
    else:
        d_x_hat = d_out * weights[None, :]
        d_mu = torch.sum(- d_x_hat * sqrt_var_inv[None,:], dim=dim)
        d_var = torch.sum(- 0.5 * d_x_hat * x_minus_mu * sqrt_var_inv[None,:]**3, dim=dim)
        d_in = d_x_hat * sqrt_var_inv[None,:] + d_mu[None,:] / bz + 2 * d_var[None, :] * (x_minus_mu) / bz
    
    return d_in, d_weights, d_bias, d_mu, d_var

def check_linear(k, cfg):
    file_head = os.path.join(root, k)
    shape_in = (bz, cfg['in_features'])
    shape_out = (bz, cfg['out_features'])
    shape_w = (cfg['out_features'], cfg['in_features'])
    # input & weights
    input = read_matrix_from_txt(file_head + '.in.txt', shape_in)
    weights = read_matrix_from_txt(file_head + '.weights.txt', shape_w)
    # gradients
    d_out = read_matrix_from_txt(file_head+'.d_out.txt', shape_out)
    d_in = read_matrix_from_txt(file_head + '.d_in.txt', shape_in)
    d_weights = read_matrix_from_txt(file_head + '.d_weights.txt', shape_w)
    d_bias = read_matrix_from_txt(file_head+'.d_bias.txt', (cfg['out_features']))

    d_in_py, d_weights_py, d_bias_py = linear_backprop(d_out, input, weights)

    r1, r2, r3 = compare(d_in, d_weights, d_bias, d_in_py, d_weights_py, d_bias_py)
    re = []
    if not r1:
        save_tensor_to_txt(d_in_py, file_head+'.d_in_py.txt')
        re.append('d_in')
    if not r2:
        save_tensor_to_txt(d_weights_py, file_head+'.d_weights_py.txt')
        re.append('d_weights')
    if not r3:
        save_tensor_to_txt(d_bias_py, file_head+'.d_bias_py.txt')
        re.append('d_bias')
    if r1 and r2 and r3:
        return None
    else:
        return re


def check_conv(k, cfg):
    file_head = os.path.join(root, k)
    shape_in = (bz, cfg['in_channels'], cs)
    shape_out = (bz, cfg['out_channels'], cs)
    shape_w = (cfg['out_channels'], cfg['in_channels'])
    # input & weights
    input = read_matrix_from_txt(file_head + '.in.txt', shape_in)
    weights = read_matrix_from_txt(file_head + '.weights.txt', shape_w)
    # gradients
    d_out = read_matrix_from_txt(file_head+'.d_out.txt', shape_out)
    d_in = read_matrix_from_txt(file_head + '.d_in.txt', shape_in)
    d_weights = read_matrix_from_txt(file_head + '.d_weights.txt', shape_w)
    d_bias = read_matrix_from_txt(file_head+'.d_bias.txt', (cfg['out_channels']))

    d_in_py, d_weights_py, d_bias_py = conv1d_backprop(d_out, input, weights)

    r1, r2, r3 = compare(d_in, d_weights, d_bias, d_in_py, d_weights_py, d_bias_py)
    re = []
    if not r1:
        save_tensor_to_txt(d_in_py, file_head+'.d_in_py.txt')
        re.append('d_in')
    if not r2:
        save_tensor_to_txt(d_weights_py, file_head+'.d_weights_py.txt')
        re.append('d_weights')
    if not r3:
        save_tensor_to_txt(d_bias_py, file_head+'.d_bias_py.txt')
        re.append('d_bias')
    if r1 and r2 and r3:
        return None
    else:
        return re


def check_bn(k, cfg):
    file_head = os.path.join(root, k)
    if cfg['dim'] == 3:
        shape_in = (bz, cfg['num_features'], cs)
        shape_out = (bz, cfg['num_features'], cs)
    else:
        shape_in = (bz, cfg['num_features'])
        shape_out = (bz, cfg['num_features'])
    shape_w = (cfg['num_features'])
    # input & weights
    x_hat = read_matrix_from_txt(file_head + '.x_hat.txt', shape_in)
    x_minus_mu = read_matrix_from_txt(file_head + '.x_minus_mu.txt', shape_in)
    weights = read_matrix_from_txt(file_head + '.weights.txt', shape_w)

    sqrt_var_inv = read_matrix_from_txt(file_head + '.sqrt_var_inv.txt', (cfg['num_features']))
    d_mu = read_matrix_from_txt(file_head + '.d_mu.txt', (cfg['num_features']))
    d_var = read_matrix_from_txt(file_head + '.d_var.txt', (cfg['num_features']))
    # gradients
    d_out = read_matrix_from_txt(file_head+'.d_out.txt', shape_out)
    d_in = read_matrix_from_txt(file_head + '.d_in.txt', shape_in)
    d_weights = read_matrix_from_txt(file_head + '.d_weights.txt', shape_w)
    d_bias = read_matrix_from_txt(file_head+'.d_bias.txt', (cfg['num_features']))

    d_in_py, d_weights_py, d_bias_py, d_mu_py, d_var_py = batchnorm1d_backprop(d_out, weights, x_hat, x_minus_mu, sqrt_var_inv)

    r1, r2, r3, r4, r5 = compare_bn(d_in, d_weights, d_bias, d_mu, d_var, d_in_py, d_weights_py, d_bias_py, d_mu_py, d_var_py)
    re = []
    if not r1:
        save_tensor_to_txt(d_in_py, file_head+'.d_in_py.txt')
        re.append('d_in')
    if not r2:
        save_tensor_to_txt(d_weights_py, file_head+'.d_weights_py.txt')
        re.append('d_weights')
    if not r3:
        save_tensor_to_txt(d_bias_py, file_head+'.d_bias_py.txt')
        re.append('d_bias')
    if not r4:
        save_tensor_to_txt(d_mu_py, file_head+'.d_mu_py.txt')
        re.append('d_mu')
    if not r5:
        save_tensor_to_txt(d_var_py, file_head+'.d_var_py.txt')
        re.append('d_var')
    if r1 and r2 and r3 and r4 and r5:
        return None
    else:
        return re
    
def check_else():
    encoder_path = os.path.join(root, 'feat')
    trans_feat = read_matrix_from_txt(encoder_path + ".trans_feat.txt", (bz, 64, 64))
    feat = read_matrix_from_txt(encoder_path + ".feat.txt", (bz, 64, cs))
    feat_d_out = read_matrix_from_txt(encoder_path + ".feat_d_out.txt", (bz, 64, cs))
    f_gradients = read_matrix_from_txt(encoder_path + ".d_feat.txt", (bz, 64, cs))
    trans_feat_gradients = read_matrix_from_txt(encoder_path + ".d_trans_feat.txt", (bz, 64, 64))
    point_d_out = read_matrix_from_txt(encoder_path + ".point_d_out.txt", (bz, 3, cs))
    input = read_matrix_from_txt(encoder_path + ".point_d_out.txt", (bz, cs, 3))
    trans_point_gradients = read_matrix_from_txt(encoder_path + ".d_trans_point.txt", (bz, 3, 3))

    f_gradients_py = torch.bmm(trans_feat, feat_d_out)
    trans_feat_gradients_py = torch.bmm(feat, feat_d_out.transpose(-2, -1))
    trans_point_gradients_py = torch.bmm(point_d_out, input)

    re = []
    r1 = compare_matrices(f_gradients, f_gradients_py)
    r2 = compare_matrices(trans_feat_gradients, trans_feat_gradients_py)
    r3 = compare_matrices(trans_point_gradients, trans_point_gradients_py)
    if not r1:
        save_tensor_to_txt(f_gradients_py, encoder_path +'.f_gradients_py.txt')
        re.append('f_gradients')
    if not r2:
        save_tensor_to_txt(trans_feat_gradients_py, encoder_path +'.d_trans_feat_py.txt')
        re.append('d_trans_feat')
    if not r3:
        save_tensor_to_txt(trans_point_gradients_py, encoder_path +'.d_trans_point_py.txt')
        re.append('d_trans_point_py')
    return re

def main():
    content = load_yaml_file(cfg_path)
    cfg_dict = parse_yaml_to_dict(content)
    error_dict = {}
    for k, v in cfg_dict.items():
        base_name = k.split('.')[-1]
        re = None
        if base_name.startswith("conv") and (target == 'conv' or target == 'all'):
            re = check_conv(k, v)
        elif base_name.startswith('bn') and (target == 'bn' or target == 'all'):
            re = check_bn(k, v)
        elif base_name.startswith('fc') and (target == 'linear' or target == 'all'):
            re = check_linear(k, v)
        
        if re is not None:
            error_dict[k]=re
    
    re = check_else()
    if re is not None:
        error_dict['else'] = re
    if len(error_dict.keys()) > 0:
        print("[ER] CHEKC GRADS:")
        for k, v in error_dict.items():
            print(k, ': ', v, sep='')
    else:
        print("[AC] UNIT TEST SUCCESSFULLY")
        


if __name__=='__main__':
    main()