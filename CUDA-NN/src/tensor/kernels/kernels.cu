#include "kernels.cuh"

__global__
void kScale(float *d_data, float factor, size_t N) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= N) return;

    d_data[tid] = d_data[tid] * factor;
}

__global__
void kSum(float *d_M, float *d_out, int N) {
    __shared__ float sd_M[BLOCK_SIZE1D];

    int tid = threadIdx.x;
    int gidx = blockIdx.x * blockDim.x + threadIdx.x;

    int block_num = gridDim.x;

    ///////////////////////////////////////////////
    // the first loop: read from d_M
    if (gidx < N) sd_M[tid] = d_M[gidx];
    else sd_M[tid] = 0;
    __syncthreads();

    for(int stride=blockDim.x/2; stride>0; stride>>=1) {
        if(tid < stride && tid + stride < N) {
            sd_M[tid] += sd_M[tid + stride];
        }
        __syncthreads();
    }
    if(tid == 0) d_out[blockIdx.x] = sd_M[0];
    
    N = block_num;
    block_num = (block_num-1)/blockDim.x + 1;
    gidx = (gidx-1) / blockDim.x + 1;
    //////////////////////////////////////////////////

    while(N >= blockDim.x) {
        if (gidx < N) 
            sd_M[tid] = d_out[gidx];
        else 
            sd_M[tid] = 0;
        __syncthreads();

        // reduce the shared memory
        for(int stride=blockDim.x/2; stride>0; stride>>=1) {
            if(tid < stride && tid + stride < N) {
                sd_M[tid] += sd_M[tid + stride];
            }
            __syncthreads();
        }

        if(tid == 0) d_out[blockIdx.x] = sd_M[0];

        N = block_num;
        block_num = (N-1)/blockDim.x + 1;
        gidx = (gidx-1) / blockDim.x + 1;
    }


    // N < BLOCK_SIZE
    if(tid < N) sd_M[tid] = d_out[tid];
    else        sd_M[tid] = 0;

    for(int stride=blockDim.x/2; stride>0; stride>>=1) {
        if(tid < stride) {
            sd_M[tid] += sd_M[tid + stride];
        }
        __syncthreads();
    }

    if (tid==0) d_out[0] = sd_M[0];
}


__global__ 
void kTranspose(float* d_M, float* d_out, int m, int n) {
    __shared__ float sd_M[BLOCK_SIZE2D][BLOCK_SIZE2D];
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= n || y >= m) {
        return;
    }

    sd_M[threadIdx.y][threadIdx.x] = d_M[y * n + x];

    __syncthreads();

    d_out[x * m + y] = sd_M[threadIdx.y][threadIdx.x];
}









