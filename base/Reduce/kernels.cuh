#define BLOCK_SIZE 32
#define NUM_PER_THREAD 4

#ifndef KERNELS_H
#define KERNELS_H

__global__ void reduceSum(float *d_M, float *d_out, int N);

__global__ void reduceSum_op1(float *d_M, float *d_out, int N);

__global__ void reduceSum_op2(float *d_M, float *d_out, int N);

__global__ void reduceSum_op3(float *d_M, float *d_out, int N);

__global__ void reduceSum_op4(float *d_M, float *d_out, int N);

__global__ void reduceSum_op5(float *d_M, float *d_out, int N);

__global__ void reduceSum_op6(float *d_M, float *d_out, int N);

#endif