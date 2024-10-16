# CUDA-NN
Some learning code, simple reproduction in python and comprehensive implementation in CUDA of neural network.

* Basic code for CUDA learner to get started.
* Simple python implementation of [PointNet](https://arxiv.org/abs/1612.00593).
* Implementation of tensor, layers and PointNet model in CUDA. Training and inference code of PointNet in CUDA.

## Basic Code for CUDA
Including 3 main parts:
* Matmul: the base version of matrix multiplication and several optimized versions.
* Reduce: versions of sum operatoin on matrix.
* Transpose: basic and optimized version of matrix transpose.

Most of optimizations concentrate on the following perspective:
* Shared memory: tiling tech, optim in w/r like avoiding discrete reading.
* Rigister memory: increase the ratio of computation and memory access to enhance FLOPS.
* Bank conflict: avoid bank conflict caused by shared memory.
* [TODO] Memory access acceleration: float4 reading ......

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

## Simple implementation of PointNet in Python
Containing code used for model impl and training of PointNet in single GPU.

To run:
```
cd CUDA-NN/python
python train.py
```

[TODO] Optimize the structure of PointNet or training to improve the accuracy.

## Implementation of PointNet in CUDA
As a big assignment for the UCAS GPU course, we were asked to use CUDA to implement PointNet inference and training, and were required to have certain performance in accuracy and speed. This might be a pefect project to practice CUDA with.

In this project, layers and sub-modules in PointNet are implemented in `CUDA-NN/src`, to run the project:
```
cd CUDA-NN/CUDA-NN
mkdir build
cd build
cmake ..
make run
```

Helpful reference:
* https://github.com/jpowie01/CUDA-DNN-MNIST


## *Other things for GPU course of UCAS*

### 1. merge the project
Versions in the course server(Ubuntu 18.04):
* nvcc: 12.1
* gcc: 7.5

The profiller only supports submitting a single `.cu` file, so it's needed to merge the entire project into a single file. To run the merge script:

```
bash merge.sh test/main # the main cuda file
```

And use official `nvcc` to compile the `merged.cu` written in `compile.sh`:
```
nvcc merged.cu -o train -Xcompiler "-O3 -std=c++14" -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_62,code=sm_62 -gencode arch=compute_70,code=sm_70 -lhdf5_cpp -lhdf5

./train
```

*PS: the compile process may fail for the version of gcc or the arch of the GPU. Please check the relevant correspondence if got error.*

### 2. program beat
To verify the correctness of the layers or models, it's a must to compare them with the official implementation by torch at the input/output level. To do this：

* Ensure absolute or relative paths used in both cuda or python programs are correct. Recommend: absolute path instead of relative path.
* Write and ensure the infomation in `config.yaml` is matched.
* Ensure `make run` successfully in the `CUDA-NN/build` directory.

Then run:
```
python beat.py
```

If all test points passed, you will get like:
```
Test(python) module POINTNET
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:00<00:00, 21.80it/s]
--------------------------------------------------
Test(cuda) module POINTNET
--------------------------------------------------
Beat the outputs
--------------------------------------------------
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:00<00:00, 2445.66it/s]
[AC] Test all 10 points successfully!
```


