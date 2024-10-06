#include "kernels.cuh"
#include "stdio.h"

__global__
void kAdd_l2(
    float *d_A, float *d_B, float *d_out, int M, int N, float f1, float f2
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    d_out[row * N + col] = f1 * d_A[row * N + col] + f2 * d_B[col];
}

/* 
Assume that s_i is N or 0 when dim1=dim2=2:
 - s1 = 0 && s2 = 0: 1D + 1D
 - s1 = N && s2 = 0: 2D + 1D
 - s1 = 0 && s2 = N: 1D + 2D
 - s1 = N && s2 = N: 2D + 2D
*/
__global__
void kAddStride_l2(
    float *d_A, float *d_B, float *d_out, int M, int N, float f1, float f2, 
    int s1, int s2
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    d_out[row * N + col] = f1 * d_A[row * s1 + col] + f2 * d_B[row * s2 + col];
}

__global__
void kMatmul_l2(
    float*d_A, float* d_B, float*d_out, int M, int N 
) {
    // A (M x N) @ B (N) = C (M)
    // __shared__ float ds_A[BLOCK_SIZE1D][TILE_SIZE];
    __shared__ float ds_B[TILE_SIZE];
    float o_gid = 0.0;
    // int tid = threadIdx.x;
    int gid = blockDim.y * blockIdx.y + threadIdx.x;

    int phase = (N-1) / TILE_SIZE + 1;

    for(int p=0; p<phase; p++) {
        for(int i=0; i<TILE_SIZE; i++) {
            ds_B[i] = d_B[p*TILE_SIZE + i];
        }
        __syncthreads();
        for(int i=0; i<TILE_SIZE; i++) {
            o_gid += d_A[gid * N + p*TILE_SIZE + i] * ds_B[i];
        }
        __syncthreads();
    }

    d_out[gid] = o_gid;
}

/* 
Assume that s_i is N or 0 when dim1=dim2=2:
 - s1 = 0 && s2 = 0: (1 x N) @ (1 x N)T = (1)
 - s1 = N && s2 = 0: (M x N) @ (1 x N)T = (M x 1)
 - s1 = 0 && s2 = N: (1 x N) @ (M x N)T = (1 x M)
 - s1 = N && s2 = N: (M x N) @ (M x N)T = (M x M)
*/
__global__
void kMatmulStride_l2(
    float*d_A, float* d_B, float*d_out, int M, int N 
) {
    // A (M x N) @ B (1 x N) = C (M)
    
}