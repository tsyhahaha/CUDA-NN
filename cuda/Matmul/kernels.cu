#include "kernels.cuh"
#include<stdio.h>

/*
***************************************************************
The device memory is arranged 1D in the logical view
resulting in the extra need of op "\""%" to extract item in A B.
***************************************************************
*/
__global__ void deviceMatmul_1D(
    float *d_A, float *d_B, float *d_C, int M, int N, int K
) {
    float cVal = 0;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row = idx / K;
    int col = idx % K;

    if(idx > M*K) return;

    for (int i=0; i<N; i++) {
        cVal += d_A[row*M + i] * d_B[i*K + col];
    }

    d_C[idx] = cVal;
}

/*
************************************************************** 
The device memory is arranged 2D in the logical view.
So that the location of item in A or B dont need to use extra
op to compute.
***************************************************************
*/
__global__ void deviceMatmul_2D(
    float *d_A, float *d_B, float *d_C, int M, int N, int K
) {
    float cVal = 0.0f;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row >= M || col >= K) return;

    for (int i=0; i<N; i++) {
        cVal += d_A[row*N + i] * d_B[i*K + col];
    }

    d_C[row*K + col] = cVal;
}

/*
************************************************************** 
using TILE technology to optimize the r/w process.
***************************************************************
*/
__global__ void deviceMatmul_2D_shared(
    float *d_A, float *d_B, float *d_C, int M, int N, int K
) {
    float cVal = 0.0f;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if(row >= M || col >= K) return;

    __shared__ float ds_A[TILE_SIZE][TILE_SIZE];
    __shared__ float ds_B[TILE_SIZE][TILE_SIZE];
    int phase = (N - 1) / TILE_SIZE + 1;
    for(int p=0; p<phase;p++) {
        // row -> y, col -> x, 2D array
        if (d_A[row*N + p*TILE_SIZE + threadIdx.x] == 0) {
            printf("d_A[%d] == 0", row*N + p*TILE_SIZE + threadIdx.x);
        }

        if (d_B[(p*TILE_SIZE + threadIdx.y)*K + col] == 0) {
            printf("d_B[%d] == 0", (p*TILE_SIZE + threadIdx.y)*K + col);
        }
        ds_A[threadIdx.y][threadIdx.x] = d_A[row*N + p*TILE_SIZE + threadIdx.x];
        ds_B[threadIdx.y][threadIdx.x] = d_B[(p*TILE_SIZE + threadIdx.y)*K + col];

        __syncthreads();
        for (int i=0; i<TILE_SIZE; i++) {
            // constant: ds_A's x , ds_B's y
            if (ds_A[threadIdx.y][i] * ds_B[i][threadIdx.x] == 0) {
                printf("ds_A[%d][%d] = %f, ds_B[%d][%d] = %f, row=%d col=%d p=%d\n", threadIdx.y, i, ds_A[threadIdx.y][i], i, threadIdx.x, ds_B[i][threadIdx.x], row, col, p);
            }
            cVal += ds_A[threadIdx.y][i] * ds_B[i][threadIdx.x];
        }
        __syncthreads();
    }

    d_C[row*K + col] = cVal;
}

/*
************************************************************** 
increase computation times per thread to hide latency of r/w,
i.e. each thread evaluates multiple matrix values through regs.
BLOCK_SIZE are divided y TIED_SIZE
***************************************************************
*/
__global__ void deviceMatmul_2D_register(
    float *d_A, float *d_B, float *d_C, int M, int N, int K
) {
    float cVal = 0.0f;
    int row = (blockIdx.y * blockDim.y + threadIdx.y) * TIED_SIZE;
    int col = (blockIdx.x * blockDim.x + threadIdx.x) * TIED_SIZE;
    float tie[TIED_SIZE][TIED_SIZE] = {0.0f};

    // assume that M & K can be mod by 4
    for (int i=0; i<TIED_SIZE; i++) {
        for (int j=0; j<TIED_SIZE; j++) {
            if(row + j > M || col + i > K) return;
            // d_A[i + row][t] * d_B[row + i][col + j]
            for (int t=0; t<N; t++) {
                tie[i][j] += d_A[(i + row) * N + t] * d_B[t * K + col + j];
            }
        }
    }
    for (int index_q = 0; index_q < TIED_SIZE; index_q++){
        for (int index_v = 0; index_v < TIED_SIZE; index_v++){
            if (row + index_q < M && col + index_v < K){
                d_C[(row + index_q) * K + col + index_v] = tie[index_q][index_v];
            }
        }
    }
}




