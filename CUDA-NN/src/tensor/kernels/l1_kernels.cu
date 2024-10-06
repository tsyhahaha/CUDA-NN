#include "kernels.cuh"


__global__
void kAdd_l1(
    float *d_A, float *d_B, float *d_out, int M, float f1, float f2
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col >= M) return;

    d_out[col] = f1*d_A[col] + f2 * d_B[col];
}

__global__
void kMatmul_l1(
    float*d_A, float* d_B, float* d_out, int N
) {
    // impl: use TILE tech by a 1D block.
    // anther impl: mul first and reduce to one value.
    __shared__ float sd_M[BLOCK_SIZE1D];

    int tid = threadIdx.x;
    int gidx = blockIdx.x * blockDim.x + threadIdx.x;
    int block_num = gridDim.x;

   ///////////////////////////////////////////////
    // the first loop
    if (gidx < N) 
        sd_M[tid] = d_A[gidx] * d_B[gidx];
    else 
        sd_M[tid] = 0;
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
}