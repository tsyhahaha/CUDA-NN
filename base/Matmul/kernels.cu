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

    if(row >= M || col >= K) {
        return; }

    for (int i=0; i<N; i++) {
        cVal += d_A[row*N + i] * d_B[i*K + col];
    }

    d_C[row*K + col] = cVal;
}

/*
************************************************************** 
using TILE technology to optimize the r/w process.
 - BLOCK_SIZE -> the threads width in one block
 - TILE_SIZE  -> the amount of mul-ops in each phase or the
                window size.
***************************************************************
*/
__global__ void deviceMatmul_2D_shared(
    float *d_A, float *d_B, float *d_C, int M, int N, int K
) {
    float cVal = 0.0f;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float ds_A[BLOCK_SIZE][TILE_SIZE];
    __shared__ float ds_B[TILE_SIZE][BLOCK_SIZE];
    int phase = (N - 1) / TILE_SIZE + 1;
    for(int p=0; p<phase;p++) {
        if (row < M && p*TILE_SIZE + threadIdx.x < N && threadIdx.x < TILE_SIZE) 
        ds_A[threadIdx.y][threadIdx.x] = d_A[row*N + p*TILE_SIZE + threadIdx.x];
        if(p*TILE_SIZE + threadIdx.y < N && col < K && threadIdx.y < TILE_SIZE)
        ds_B[threadIdx.y][threadIdx.x] = d_B[(p*TILE_SIZE + threadIdx.y)*K + col];

        __syncthreads();
        for (int i=0; i<TILE_SIZE; i++) {
            // constant: ds_A's x , ds_B's y
            cVal += ds_A[threadIdx.y][i] * ds_B[i][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < K)
        d_C[row*K + col] = cVal;
}

/*
************************************************************** 
increase computation times per thread to hide latency of r/w,
i.e. each thread evaluates multiple matrix values through regs.
BLOCK_SIZE need to divide by TIED_SIZE, but if you don't, it's okay, except
wasting some threads resources in a block.
***************************************************************
*/
__global__ void deviceMatmul_2D_register(
    float *d_A, float *d_B, float *d_C, int M, int N, int K
) {
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

/*
************************************************************** 
using both TILE technology(shared mem) and register to 
optimize the r/w process. 
1st version: Integrate all the memory r/w involved in the 
computation into this thread
***************************************************************
*/
__global__ void deviceMatmul_2D_shared_register_1st(
    float *d_A, float *d_B, float *d_C, int M, int N, int K
) {
    int row = (blockIdx.y * blockDim.y + threadIdx.y) * TIED_SIZE;
    int col = (blockIdx.x * blockDim.x + threadIdx.x) * TIED_SIZE;

    // shared memory need to expand TIED_SIZE times on each dim
    __shared__ float ds_A[BLOCK_SIZE * TIED_SIZE][TILE_SIZE];
    __shared__ float ds_B[TILE_SIZE][BLOCK_SIZE * TIED_SIZE];
    int phase = (N - 1) / TILE_SIZE + 1;
    // register array
    float tie[TIED_SIZE][TIED_SIZE] = {0.0f};

    for(int p=0; p<phase;p++) {
        for (int i=0; i<TIED_SIZE; i++) {
            for(int j=0; j<TILE_SIZE; j++) {
                if ((row + i) < M && (p*TILE_SIZE + j) < N)
                    ds_A[threadIdx.y * TIED_SIZE + i][j] = d_A[(row + i) * N + p * TILE_SIZE + j];

                if ((p*TILE_SIZE + j) < N && (col + i) < K)
                    ds_B[j][threadIdx.x * TIED_SIZE + i] = d_B[(p*TILE_SIZE + j) * K + col + i];
            }
        }
        __syncthreads();

        for (int i=0; i<TIED_SIZE; i++) {
            for(int j=0; j<TIED_SIZE; j++) {
                for(int k=0; k<TILE_SIZE; k++)
                    tie[i][j] += ds_A[threadIdx.y * TIED_SIZE + i][k] * ds_B[k][threadIdx.x * TIED_SIZE + j];
            }
        }
        __syncthreads();
    }
    
    // write tied result to d_C
    for(int i=0; i<TIED_SIZE; i++) {
        for (int j=0; j<TIED_SIZE; j++) {
            if (row + i < M && col + j < K) {
                d_C[(row+i) * K + col + j] = tie[i][j];
            }
        }
    }
}

/*
************************************************************** 
using both TILE technology(shared mem) and register to 
optimize the r/w process. 
2nd version: Use the saved threads within the block for 
memory r/w rather than just use computation threads.
***************************************************************
*/
__global__ void deviceMatmul_2D_shared_register_2nd(
    float *d_A, float *d_B, float *d_C, int M, int N, int K
) {
    int row = (blockIdx.y * blockDim.y + threadIdx.y) * TIED_SIZE;
    int col = (blockIdx.x * blockDim.x + threadIdx.x) * TIED_SIZE;

    // shared memory need to expand TIED_SIZE times on each dim
    __shared__ float ds_A[BLOCK_SIZE * TIED_SIZE][TILE_SIZE];
    __shared__ float ds_B[TILE_SIZE][BLOCK_SIZE * TIED_SIZE];
    int phase = (N - 1) / TILE_SIZE + 1;
    // register array
    float tie[TIED_SIZE][TIED_SIZE] = {0.0f};

    int id = threadIdx.x + threadIdx.y * blockDim.x;
    int ds_row = id / BLOCK_SIZE, ds_col = id % BLOCK_SIZE; // assume that TILE_SIZE == BLOCK_SIZE

    for(int p=0; p<phase;p++) {
        // Read the data with some other thread
        if (row + TIED_SIZE - 1 < M && ds_col + p*TILE_SIZE < N &&
            TIED_SIZE*ds_row+TIED_SIZE -1 < BLOCK_SIZE * TIED_SIZE && 
            ds_col < TILE_SIZE) {
            for (int i = 0; i < TILE_SIZE; i++) 
                ds_A[TIED_SIZE * ds_row + i][ds_col] = d_A[(row + i) * N + ds_col + p * TILE_SIZE];
        }
        if (ds_row + p * TILE_SIZE < N && col + TIED_SIZE - 1 < K &&
            TIED_SIZE * ds_col + TIED_SIZE -1 < BLOCK_SIZE * TIED_SIZE &&
            ds_row < TILE_SIZE
        ) {
            for (int i=0; i<TILE_SIZE; i++) 
                ds_B[ds_row][TIED_SIZE * ds_col + i] = d_B[(ds_row + p * TILE_SIZE) * K + col + i];
        }

        __syncthreads();

        for (int i=0; i<TIED_SIZE; i++) {
            for(int j=0; j<TIED_SIZE; j++) {
                for(int k=0; k<TILE_SIZE; k++)
                    tie[i][j] += ds_A[threadIdx.y * TIED_SIZE + i][k] * ds_B[k][threadIdx.x * TIED_SIZE + j];
            }
        }
        __syncthreads();
    }
    
    // write tied result to d_C
    for(int i=0; i<TIED_SIZE; i++) {
        for (int j=0; j<TIED_SIZE; j++) {
            if (row + i < M && col + j < K) {
                d_C[(row+i) * K + col + j] = tie[i][j];
            }
        }
    }
}

/*
***************************************************************
using float4 type to optimize the w/r velocity.
***************************************************************
*/
__global__ void deviceMatmul_2D_shared_register_float4_1st(
    float *d_A, float *d_B, float *d_C, int M, int N, int K
) {
    int row = (blockIdx.y * blockDim.y + threadIdx.y) * TIED_SIZE;
    int col = (blockIdx.x * blockDim.x + threadIdx.x) * TIED_SIZE;

    // shared memory need to expand TIED_SIZE times on each dim
    __shared__ float ds_A[BLOCK_SIZE * TIED_SIZE][TILE_SIZE];
    __shared__ float ds_B[TILE_SIZE][BLOCK_SIZE * TIED_SIZE];
    int phase = (N - 1) / TILE_SIZE + 1;
    // register array
    float tie[TIED_SIZE][TIED_SIZE] = {0.0f};

    int id = threadIdx.x + threadIdx.y * blockDim.x;
    int ds_row = id / BLOCK_SIZE, ds_col = id % BLOCK_SIZE; // assume that TILE_SIZE == BLOCK_SIZE

    for(int p=0; p<phase;p++) {
        // Read the data with some other thread
        if (row + TIED_SIZE - 1 < M && ds_col + p*TILE_SIZE < N &&
            TIED_SIZE*ds_row+TIED_SIZE -1 < BLOCK_SIZE * TIED_SIZE && 
            ds_col < TILE_SIZE) {
            for (int i = 0; i < TILE_SIZE; i++) 
                ds_A[TIED_SIZE * ds_row + i][ds_col] = d_A[(row + i) * N + ds_col + p * TILE_SIZE];
        }
        if (ds_row + p * TILE_SIZE < N && col + TIED_SIZE - 1 < K &&
            TIED_SIZE * ds_col + TIED_SIZE -1 < BLOCK_SIZE * TIED_SIZE &&
            ds_row < TILE_SIZE
        ) {
            for (int i=0; i<TILE_SIZE; i++) 
                ds_B[ds_row][TIED_SIZE * ds_col + i] = d_B[(ds_row + p * TILE_SIZE) * K + col + i];
        }

        __syncthreads();

        for (int i=0; i<TIED_SIZE; i++) {
            for(int j=0; j<TIED_SIZE; j++) {
                for(int k=0; k<TILE_SIZE; k++)
                    tie[i][j] += ds_A[threadIdx.y * TIED_SIZE + i][k] * ds_B[k][threadIdx.x * TIED_SIZE + j];
            }
        }
        __syncthreads();
    }
    
    // write tied result to d_C
    for(int i=0; i<TIED_SIZE; i++) {
        for (int j=0; j<TIED_SIZE; j++) {
            if (row + i < M && col + j < K) {
                d_C[(row+i) * K + col + j] = tie[i][j];
            }
        }
    }
}
