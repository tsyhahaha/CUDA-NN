#include "kernels.cuh"
#include <stdio.h>

/*
*****************************************************
the base version:
    ~ thread(x, y)    read   d_M(y, x), 
      thread(x+1, y)  read   d_M(y, x+1)
    the step size  is N.
    ~ GPU read global mem by segments (32byte)
    ~ hard to coalesced when N is large enough 
    within a warp.
*****************************************************
*/
__global__ void transpose(float* d_M, float* d_out, int m, int n) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= m && y >= n) return;

    d_out[x * m + y] = d_M[y * n + x];
}


/*
*****************************************************
use shared mem:
    ~ read from global contiguously
    ~ the mechanism of shared mem diffs from GM.
    ~ uncontiguous reading in shared mem doesnt matter.
*****************************************************
*/
__global__ void transpose_shared(float* d_M, float* d_out, int m, int n) {
    __shared__ float sd_M[BLOCK_SIZE][BLOCK_SIZE];
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= m && y >= n) return;

    sd_M[threadIdx.y][threadIdx.x] = d_M[y * n + x];

    __syncthreads();

    d_out[x * m + y] = sd_M[threadIdx.y][threadIdx.x];
}


