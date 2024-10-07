#include "kernels.cuh"
#include "stdio.h"

__global__
void kAdd_l3(
    float *d_A, float *d_B, float *d_out, int M, int N, float f1, float f2
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    d_out[row * N + col] = f1 * d_A[row * N + col] + f2 * d_B[row * N + col];
}

__global__
void kMatmul_l3(
    float *d_A, float *d_B, float *d_out, int M, int N, int K
) {
    float cVal = 0.0f;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float ds_A[BLOCK_SIZE2D][TILE_SIZE];
    __shared__ float ds_B[TILE_SIZE][BLOCK_SIZE2D];

    int phase = (N - 1) / TILE_SIZE + 1;
    for(int p=0; p<phase;p++) {
        if (row < M && p*TILE_SIZE + threadIdx.x < N && threadIdx.x < TILE_SIZE) {
            ds_A[threadIdx.y][threadIdx.x] = d_A[row*N + p*TILE_SIZE + threadIdx.x];
        } else if(threadIdx.y < BLOCK_SIZE2D && threadIdx.x < TILE_SIZE) {
            // PS: It's faster  if TILE_SIZE is a factor of the matrix dimension
            ds_A[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if(p*TILE_SIZE + threadIdx.y < N && col < K && threadIdx.y < TILE_SIZE) {
            ds_B[threadIdx.y][threadIdx.x] = d_B[(p*TILE_SIZE + threadIdx.y)*K + col];
        } else if(threadIdx.y < TILE_SIZE && threadIdx.x < BLOCK_SIZE2D) {
            ds_B[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();
        for (int i=0; i<TILE_SIZE; i++) {
            // constant: ds_A's x , ds_B's y
            cVal += ds_A[threadIdx.y][i] * ds_B[i][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < K)
        d_out[row*K + col] = cVal;
}

__global__
void kMatmulTransposed_l3(
    float *d_A, float *d_B, float *d_out, int M, int N, int K
) {
    float cVal = 0.0f;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row >= M || col >= K) return;

    __shared__ float ds_A[BLOCK_SIZE2D][TILE_SIZE];
    __shared__ float ds_B[BLOCK_SIZE2D][TILE_SIZE];

    int phase = (N - 1) / TILE_SIZE + 1;
    for(int p=0; p<phase;p++) { 
        if (row < M && p*TILE_SIZE + threadIdx.x < N && threadIdx.x < TILE_SIZE) {
            // ds_A[ty][tx] = d_A[row][p*TILE_SIZE + tx]
            ds_A[threadIdx.y][threadIdx.x] = d_A[row*N + p*TILE_SIZE + threadIdx.x];
        } else if(threadIdx.y < BLOCK_SIZE2D && threadIdx.x < TILE_SIZE) {
            ds_A[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        if (col < K && p*TILE_SIZE + threadIdx.y < N && threadIdx.y < TILE_SIZE) {
            // ds_B[tx][ty] = d_B[col][p*TILE_SIZE + ty]
            ds_B[threadIdx.x][threadIdx.y] = d_B[col*N + p*TILE_SIZE + threadIdx.y]; 
        } else if(threadIdx.x < BLOCK_SIZE2D && threadIdx.y < TILE_SIZE) {
            ds_B[threadIdx.x][threadIdx.y] = 0.0f;
        }

        __syncthreads();
        for (int i=0; i<TILE_SIZE; i++) {
            // constant: ds_A's x , ds_B's x
            cVal += ds_A[threadIdx.y][i] * ds_B[threadIdx.x][i];
        }
        __syncthreads();
    }

    // printf("M=%d N=%d K=%d, d_out[%d][%d] = %f\n", M, N, K, row, col, cVal);
    d_out[row*K + col] = cVal;
}

__global__
void kMatmulStride_l3(
    float*d_A, float* d_B, float*d_out, int M, int N 
) {
    // A (B x M x N) @ B (B/1 x N x K) = C (B x M x K)
          
}