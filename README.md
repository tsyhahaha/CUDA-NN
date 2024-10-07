# CUDA-NN
some learning code, simple reproduction and complex implementation about neural network

* Code for CUDA learner to get started.
* Simple implementation of [PointNet](https://arxiv.org/abs/1612.00593).
* Implementation of tensor, layers and PointNet in CUDA. Training and inference code of PointNet in CUDA.

## Basic Code for CUDA
Including 3 main parts:
* Matmul: the base version of matrix multiplication and several optimized version.
* Reduce: versions of matrix sum operatoin.
* Transpose: Basic and optimized version of matrix transpose.

Most of optimization are concentrated on the following perspective:
* Shared memory: tiling tech, avoid discrete reading and other aspects of optim in w/r.
* Rigister memory: increase the ratio of computation and memory access to enhance FLOPS.
* Bank conflict: avoid bank conflict which caused by reading mode of shared memory.
* [TODO] Memory access acceleration: float4 reading.

To run the start code:
```
cd CUDA-NN/base/Matmul
nvcc kernels.cu timer.cu -o timer
./timer
```

Helpful reference:
* [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomic-functions)
* General-Purpose Graphics Processor Architecture
* [矩阵乘法的 CUDA 实现、优化及性能分析](https://chiemon.github.io/2020/02/06/CUDA-%E7%9F%A9%E9%98%B5%E4%B9%98%E6%B3%95-%E4%BC%98%E5%8C%96%E5%8F%8A%E6%80%A7%E8%83%BD%E5%88%86%E6%9E%90-%E4%B8%8A.html)
* [CUDA 矩阵乘法终极优化指南](https://zhuanlan.zhihu.com/p/410278370)
* [CUDA 矩阵转置优化](https://code.hitori.moe/post/cuda-transpose-optimization/)
* [how-to-optim-algorithm-in-cuda](https://github.com/BBuf/how-to-optim-algorithm-in-cuda)

## Simple implementation of PointNet
Containing code used for PointNet's training in single GPU.

To run:
```
cd CUDA-NN/python
python train.py
```

[TODO] Optimize the structure of PointNet||training to improve the accuracy.

## [TODO]Implementation of NN in CUDA
As a big assignment for the UCAS GPU course, we were asked to use CUDA to implement PointNet inference and training, and were required to have certain performance in accuracy and speed. This might be a pefect project to practice CUDA with.

To run the project:
```
cd CUDA-NN/CUDA-NN
mkdir build
cd build
cmake ..
make run
```

Helpful reference:
* https://github.com/jpowie01/CUDA-DNN-MNIST
* https://github.com/DeMoriarty/custom_matmul_kernels







