import torch
import torch.nn as nn
import pdb

channel = 4

bn = nn.BatchNorm1d(channel)
bn.train()
weights = bn.weight.data
bias = bn.bias.data
input = torch.rand(1, channel, 3)

mu = torch.mean(input, 0, keepdim=True)
mu = torch.mean(mu, 2, keepdim=True)
var = torch.var(input, (0,2), unbiased=False, keepdim=True)

x_mu = (input - mu)**2
print(x_mu.shape)
var_s = torch.mean(x_mu, 0, keepdim=True)
var_s = torch.mean(var_s,2, keepdim=True)

print(var_s)
print(var)



