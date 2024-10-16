# Layer Implementation Details

According to `model_struc.md`, layers to be implemented include: Linear, Conv1d, BatchNorm1d, MaxPool, ReLU, Sigmoid, Dropout.

To test these, the test program needs to be adapted first.

## Linear
accelerate: GEMM
bmm: https://developer.nvidia.com/blog/cublas-strided-batched-matrix-multiply/

In addition, there are many batch operations that can be optimized.

## Conv1d
accelerate: Matmul

## BatchNorm1d
merge to a single kernel

## Softmax
Fused softmax
maybe triton learning can get benefit.
* https://github.com/facebookincubator/AITemplate/wiki/How-to-write-a-fast-Softmax-CUDA-kernel%3F

## *Other*
* set BLOCK_SIZE or TILE_SIZE for each layer
* merge kernels like relu
* tensor core?

