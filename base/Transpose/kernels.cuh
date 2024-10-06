#define BLOCK_SIZE 8

#ifndef KERNELS_H
#define KERNELS_H

__global__ void transpose(float* d_M, float* d_out, int m, int n);

__global__ void transpose_shared(float* d_M, float* d_out, int m, int n);

#endif