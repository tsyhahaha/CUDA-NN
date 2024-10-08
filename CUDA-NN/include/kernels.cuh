#pragma once

#ifndef KERNEL_H
#define KERNEL_H

#include <assert.h>
#include <stdio.h>
#include "utils.cuh"
#include "configure.cuh"

/* Layer specific kernels*/

__global__
void kReLu1D(float* A, float* d_out, int N);

__global__
void kReLu2D(float* A, float* d_out, int M, int N);

/* Tensor specific */

__global__
void kScale(float *d_data, float factor, float offset, size_t N);

__global__ 
void kSum(float *d_M, float *d_out, int N);

__global__ 
void kTranspose(float* d_M, float* d_out, int m, int n);

/* 1 Level BLAS */
__global__
void kAdd_l1(
    float *d_A, float *d_B, float *d_out, int M, float f1, float f2
);

__global__
void kMatmul_l1(
    float*d_A, float* d_B, float* d_out, int N
);

/* 2 Level BLAS */
__global__
void kAdd_l2(
    float *d_A, float *d_B, float *d_out, int M, int N, float f1, float f2
);

__global__
void kMatmul_l2(
    float*d_A, float* d_B, float*d_out, int M, int N 
);

__global__
void kAddStride_l2(
    float *d_A, float *d_B, float *d_out, int M, int N, float f1, float f2, 
    int s1, int s2
);

/* 3 Level BLAS */
__global__
void kAdd_l3(
    float *d_A, float *d_B, float *d_out, int M, int N, float f1, float f2
);

__global__
void kMatmul_l3(
    float *d_A, float *d_B, float *d_out, int M, int N, int K
);

__global__
void kMatmulTransposed_l3(
    float *d_A, float *d_B, float *d_out, int M, int N, int K
);

#endif