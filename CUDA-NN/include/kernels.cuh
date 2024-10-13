#pragma once

#ifndef KERNEL_H
#define KERNEL_H

#include <assert.h>
#include <stdio.h>
#include "utils.cuh"
#include "configure.cuh"

#include <cuda.h>
#include "cuda_runtime.h"

/* Layer specific kernels*/

__global__
void kDropout1D(float* A, float* d_out, int N);

__global__
void kReLu1D(float* A, float* d_out, int N);

__global__
void kReLu2D(float* A, float* d_out, int M, int N);

__global__ 
void kBn1d_l2(float* d_data, float* d_out, float* weights, float* bias, 
    float* mean, float* var, float eps, int N, int C
);

__global__ 
void kBn1d_l3(float* d_data, float* d_out, float* weights, float* bias, 
    float* mean, float* var, float eps, int N, int C, int L
);

__global__
void kConv1d(float* d_in, float* d_out, float* weights, float* bias, int C_in, int L, int C_out, int N);

/* unary op */

__global__
void kSumLastDim2D(float* d_data, float* d_out, size_t C, size_t L
);

__global__
void kMaxLastDim3D(float* d_data, float* d_out, size_t N, size_t C, size_t L);

__global__
void kMaxLastDim2D(float* d_data, float* d_out, size_t C, size_t L
);

__global__
void kScale(float *d_data, float factor, float offset, size_t N);

__global__ 
void kSum(float *d_M, float *d_out, int N);

__global__ 
void kTranspose(float* d_M, float* d_out, int m, int n);

__global__
void kTransposeLast3D(float* d_data, float* d_out, size_t N, size_t m, size_t n
);

__global__
void kExp(float* d_data, float* d_out, int n_data);

/* add kernels */
__global__
void kAdd_l1(
    float *d_A, float *d_B, float *d_out, int M, float f1, float f2
);

__global__
void kAdd_l2(
    float *d_A, float *d_B, float *d_out, int M, int N, float f1, float f2
);

__global__
void kAdd_l3(
    float *d_A, float *d_B, float *d_out, int M, int N, float f1, float f2
);

__global__
void kAddStride_l1(
    float *d_A, float *d_B, float *d_out, int M, 
    float f1, float f2, int s1, int s2
);

__global__
void kAddStride_l2(
    float *d_A, float *d_B, float *d_out, int M, int N, float f1, float f2, 
    int s11, int s12, int s21, int s22
);

__global__
void kAddStride_l3(
    float *d_A, float *d_B, float *d_out, int B, int M, int N, float f1, float f2,
    int s11, int s12, int s13, int s21, int s22, int s23
);

/* dotmul kernels */

__global__
void kDotStride_l1(
    float *d_A, float *d_B, float *d_out, int M, float f,
    int s1, int s2
);

__global__
void kDotStride_l2(
    float *d_A, float *d_B, float *d_out, int M, int N, float f, 
    int s11, int s12, int s21, int s22
);

__global__
void kDotStride_l3(
    float *d_A, float *d_B, float *d_out, int B, int M, int N, float f, 
    int s11, int s12, int s13, int s21, int s22, int s23
);

/* div kernels */

__global__
void kDivStride_l1(
    float *d_A, float *d_B, float *d_out, int M, float f,
    int s1, int s2
);

__global__
void kDivStride_l2(
    float *d_A, float *d_B, float *d_out, int M, int N, float f, 
    int s11, int s12, int s21, int s22
);

__global__
void kDivStride_l3(
    float *d_A, float *d_B, float *d_out, int B, int M, int N, float f, 
    int s11, int s12, int s13, int s21, int s22, int s23
);

/* matmul kernels */


__global__
void kMatmul_l1(
    float*d_A, float* d_B, float* d_out, int N
);

__global__
void kMatmul_l2(
    float*d_A, float* d_B, float*d_out, int M, int N 
);

__global__
void kMatmul_l3(
    float *d_A, float *d_B, float *d_out, int M, int N, int K
);

__global__
void kMatmulTransposed_l3(
    float *d_A, float *d_B, float *d_out, int M, int N, int K
);

__global__
void kBatchMatmul3D(
    float*d_A, float* d_B, float*d_out, int B, int M, int N, int K
);

#endif /* !KERNELS_H */