#pragma once

#ifndef KERNEL_H
#define KERNEL_H

#include <assert.h>
#include <stdio.h>
#include "utils.cuh"

#define BLOCK_SIZE1D 16

#define BLOCK_SIZE2D 4  
// less than the minimum value of the column and column of the matrix involved in the calculation
#define TILE_SIZE 2

/* Tensor specific */

__global__
void kScale(float *d_data, float factor, size_t N);

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