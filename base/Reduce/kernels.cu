#include<stdio.h>
#include "kernels.cuh"

/*
**************************************************
The simplest reduce: divide and conquer.
**************************************************
*/
__global__ void reduceSum(float *d_M, float *d_out, int N) {
    __shared__ float sd_M[BLOCK_SIZE];  // 1D assumption

    const unsigned int tid = threadIdx.x;

    // read from Vector, global -> shared
    // printf("sd_M[%d] = %f\n", tid, d_M[blockIdx.x * blockDim.x + threadIdx.x]);
    sd_M[tid] = d_M[blockIdx.x * blockDim.x + threadIdx.x];
    __syncthreads();

    // reduce process
    for(int stride=1; stride<blockDim.x; stride*=2) {
        if(tid % (2 * stride) == 0 && tid + stride < N) {
            sd_M[tid] += sd_M[tid + stride];
        }
        __syncthreads();
    }

    // Each block: reduce to the first element
    if(tid == 0)
        d_out[blockIdx.x] = sd_M[0];

}

/*
**************************************************
use interleaved addressing so that the thread 
index involved in the computation are continuous,
reducing warp divergence.
**************************************************
*/
__global__ void reduceSum_op1(float *d_M, float *d_out, int N) {
    __shared__ float sd_M[BLOCK_SIZE];  // 1D assumption

    const unsigned int tid = threadIdx.x;

    // read from Vector, global -> shared
    sd_M[tid] = d_M[blockIdx.x * blockDim.x + threadIdx.x];
    __syncthreads();

    // reduce process
    for(int stride=1; stride<blockDim.x; stride<<=1) {
        int index = 2 * stride * tid;
        if(index < blockDim.x && tid + stride < N) {
            sd_M[index] += sd_M[index + stride];
        }
        __syncthreads();
    }

    // Each block: reduce to the first element
    if(tid == 0)
        d_out[blockIdx.x] = sd_M[0];

}

/*
**************************************************
Sequential addressing.
change the r/w of shared mem to avoid bank conflict.
    ~ ds are divide into 32 banks
    ~ w/r shared mem by warp(32 threads)
    ~ data in ds are arraged to banks by 32-bit gap:
      ds_f[ 0]-bank[ 0]
      ds_f[ 1]-bank[ 1]
      ds_f[ 2]-bank[ 2]
      ...
      ds_f[31]-bank[31]
      ds_f[32]-bank[ 0]
    ~ each bank has a bandwidth of 32 bits/clock cycle
    ~ w/r the same bank in a warp raise bank conflict
**************************************************
*/
__global__ void reduceSum_op2(float *d_M, float *d_out, int N) {
    __shared__ float sd_M[BLOCK_SIZE];  // 1D assumption

    const unsigned int tid = threadIdx.x;

    // read from Vector, global -> shared
    sd_M[tid] = d_M[blockIdx.x * blockDim.x + threadIdx.x];
    __syncthreads();

    // reduce process
    for(int stride=blockDim.x/2; stride>0; stride>>=1) {
        if(tid < stride && tid + stride < N) {
            sd_M[tid] += sd_M[tid + stride];
        }
        __syncthreads();
    }

    // Each block: reduce to the first element
    if(tid == 0)
        d_out[blockIdx.x] = sd_M[0];

}

/*
**************************************************
Force idle threads to perform a computation.
    ~ decrease the usage of block(grid_dim /= 2)
**************************************************
*/
__global__ void reduceSum_op3(float *d_M, float *d_out, int N) {
    __shared__ float sd_M[BLOCK_SIZE];  // 1D assumption

    const unsigned int tid = threadIdx.x;

    const unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    // read from Vector, global -> shared
    // sd_M[tid] = d_M[blockIdx.x * blockDim.x + threadIdx.x];
    if (i + blockDim.x < N)
        sd_M[tid] = d_M[i] + d_M[i + blockDim.x];
    else
        sd_M[tid] = d_M[i];
    __syncthreads();

    // reduce process
    for(int stride=blockDim.x/2; stride>0; stride>>=1) {
        if(tid < stride && tid + stride < N) {
            sd_M[tid] += sd_M[tid + stride];
        }
        __syncthreads();
    }

    // Each block: reduce to the first element
    if(tid == 0)
        d_out[blockIdx.x] = sd_M[0];

}

/*
**************************************************
unroll the last warp.
Synchronization can be eliminated when there is 
only one active warp.
    ~ The 32 threads in a warp are synchronized
**************************************************
*/
__device__ void warpReduce(volatile float* cache, unsigned int tid){
    cache[tid]+=cache[tid+32];
    cache[tid]+=cache[tid+16];
    cache[tid]+=cache[tid+8];
    cache[tid]+=cache[tid+4];
    cache[tid]+=cache[tid+2];
    cache[tid]+=cache[tid+1];
}
__global__ void reduceSum_op4(float *d_M, float *d_out, int N) {
    __shared__ float sd_M[BLOCK_SIZE];  // 1D assumption

    const unsigned int tid = threadIdx.x;

    const unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    // read from Vector, global -> shared
    // sd_M[tid] = d_M[blockIdx.x * blockDim.x + threadIdx.x];
    if (i + blockDim.x < N)
        sd_M[tid] = d_M[i] + d_M[i + blockDim.x];
    else
        sd_M[tid] = d_M[i];
    __syncthreads();

    // reduce process
    for(int stride=blockDim.x/2; stride>32; stride>>=1) {
        if(tid < stride && tid + stride < N) {
            sd_M[tid] += sd_M[tid + stride];
        }
        __syncthreads();
    }

    // Each block: reduce to the first element
    if(tid < 32) warpReduce(sd_M, tid);
    if(tid == 0)
        d_out[blockIdx.x] = sd_M[0];
}

/*
**************************************************
unroll the last warp.
Synchronization can be eliminated when there is 
only one active warp.
    ~ The 32 threads in a warp are synchronized
**************************************************
*/
__device__ void warpReduce_op5(volatile float* cache, unsigned int tid){
    if(BLOCK_SIZE >= 64)cache[tid]+=cache[tid+32];
    if(BLOCK_SIZE >= 32)cache[tid]+=cache[tid+16];
    if(BLOCK_SIZE >= 16)cache[tid]+=cache[tid+8];
    if(BLOCK_SIZE >= 8)cache[tid]+=cache[tid+4];
    if(BLOCK_SIZE >= 4)cache[tid]+=cache[tid+2];
    if(BLOCK_SIZE >= 2)cache[tid]+=cache[tid+1];
}
__global__ void reduceSum_op5(float *d_M, float *d_out, int N) {
    __shared__ float sd_M[BLOCK_SIZE];  // 1D assumption

    const unsigned int tid = threadIdx.x;

    const unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    // read from Vector, global -> shared
    if (i + blockDim.x < N)
        sd_M[tid] = d_M[i] + d_M[i + blockDim.x];
    else
        sd_M[tid] = d_M[i];
    __syncthreads();

    // reduce process
    // do reduction in shared mem
    if(BLOCK_SIZE>=512){
        if(tid<256){
            sd_M[tid]+=sd_M[tid+256];
        }
        __syncthreads();
    }
    if(BLOCK_SIZE>=256){
        if(tid<128){
            sd_M[tid]+=sd_M[tid+128];
        }
        __syncthreads();
    }
    if(BLOCK_SIZE>=128){
        if(tid<64){
            sd_M[tid]+=sd_M[tid+64];
        }
        __syncthreads();
    }

    // Each block: reduce to the first element
    if(tid < 32) warpReduce_op5(sd_M, tid);
    if(tid == 0)
        d_out[blockIdx.x] = sd_M[0];
}


/*
**************************************************
unroll the last warp.
Synchronization can be eliminated when there is 
only one active warp.
    ~ The 32 threads in a warp are synchronized
**************************************************
*/
__global__ void reduceSum_op6(float *d_M, float *d_out, int N) {
    __shared__ float sd_M[BLOCK_SIZE];  // 1D assumption

    const unsigned int tid = threadIdx.x;

    const unsigned int i = blockIdx.x * (blockDim.x * NUM_PER_THREAD) + threadIdx.x;

    sd_M[tid] = 0;

    // read from Vector, global -> shared
    for(int iter=0; iter<NUM_PER_THREAD; iter++){
        sd_M[tid] += d_M[i+iter*BLOCK_SIZE];
    }
    __syncthreads();

    // reduce process
    // do reduction in shared mem
    if(BLOCK_SIZE>=512){
        if(tid<256){
            sd_M[tid]+=sd_M[tid+256];
        }
        __syncthreads();
    }
    if(BLOCK_SIZE>=256){
        if(tid<128){
            sd_M[tid]+=sd_M[tid+128];
        }
        __syncthreads();
    }
    if(BLOCK_SIZE>=128){
        if(tid<64){
            sd_M[tid]+=sd_M[tid+64];
        }
        __syncthreads();
    }

    // Each block: reduce to the first element
    if(tid < 32) warpReduce_op5(sd_M, tid);
    if(tid == 0)
        d_out[blockIdx.x] = sd_M[0];
}