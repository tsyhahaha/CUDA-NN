#include "kernels.cuh"
#include "configure.cuh"


__global__
void kMaxLastDim3D(float* d_data, float* d_out, size_t N, size_t C, size_t L
) {
    // It'll be faster if blocksize is the factor of L.
    int x = blockIdx.x;
    int y = blockIdx.y;
    int tid = threadIdx.x;

    // if(x >= C || y >= N) return;

    __shared__ float sd_M[BLOCK_SIZE1D];
    float cur_max = 0.0f;

    int iter = (L-1)/BLOCK_SIZE1D + 1;
    for(int i=0; i<iter; i++) {
        if (i*BLOCK_SIZE1D + tid < L) {
            sd_M[tid] = d_data[(y*C*L + x*L) + i*BLOCK_SIZE1D + tid];
        }
        __syncthreads();

        // reduce max and save to `cur_max`
        for(int stride=blockDim.x/2; stride>0; stride>>=1) {
            if(tid < stride && tid + stride + i*BLOCK_SIZE1D < L) {
                sd_M[tid] = sd_M[tid] > sd_M[tid + stride]? sd_M[tid] : sd_M[tid+stride];
            }
            __syncthreads();
        }
        cur_max = cur_max >= sd_M[0] ? cur_max : sd_M[0];
    }


    if (tid==0 && y < N && x < C)
        // printf("d_out[%d][%d] = %f\n",y, x, cur_max);
        d_out[y * C + x] = cur_max;
}

__global__
void kMaxLastDim2D(float* d_data, float* d_out, size_t C, size_t L
) {
    // It'll be faster if blocksize is the factor of L.
    int x = blockIdx.x;
    int tid = threadIdx.x;

    __shared__ float sd_M[BLOCK_SIZE1D];
    float cur_max = 0.0f;

    int iter = (L-1)/BLOCK_SIZE1D + 1;
    for(int i=0; i<iter; i++) {
        if (i*BLOCK_SIZE1D + tid < L) {
            sd_M[tid] = d_data[x*L + i*BLOCK_SIZE1D + tid];
        }
        __syncthreads();

        // reduce max and save to `cur_max`
        for(int stride=blockDim.x/2; stride>0; stride>>=1) {
            if(tid < stride && tid + stride + i*BLOCK_SIZE1D < L) {
                sd_M[tid] = sd_M[tid] > sd_M[tid + stride]? sd_M[tid] : sd_M[tid+stride];
            }
            __syncthreads();
        }
        cur_max = cur_max >= sd_M[0] ? cur_max : sd_M[0];
    }


    if (tid==0 && x < C)
        d_out[x] = cur_max;
}

__global__
void kSumLastDim2D(float* d_data, float* d_out, size_t C, size_t L
) {
    // It'll be faster if blocksize is the factor of L.
    int x = blockIdx.x;
    int tid = threadIdx.x;

    __shared__ float sd_M[BLOCK_SIZE1D];
    float cVal = 0.0f;

    int iter = (L-1)/BLOCK_SIZE1D + 1;
    for(int i=0; i<iter; i++) {
        if (i*BLOCK_SIZE1D + tid < L) {
            sd_M[tid] = d_data[x*L + i*BLOCK_SIZE1D + tid];
        }
        __syncthreads();

        for(int stride=blockDim.x/2; stride>0; stride>>=1) {
            if(tid < stride && tid + stride + i*BLOCK_SIZE1D < L) {
                sd_M[tid] = sd_M[tid] + sd_M[tid + stride];
            }
            __syncthreads();
        }
        cVal =  cVal + sd_M[0];
    }


    if (tid==0 && x < C)
        d_out[x] = cVal;
}

__global__
void kTransposeLast3D(float* d_data, float* d_out, size_t N, size_t row, size_t col
){
    __shared__ float sd_M[BLOCK_SIZE2D][BLOCK_SIZE2D];

    for(int b=0; b<N; b++) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int stride = b*row*col;

        // if(y < row && x < col)
        //     d_out[stride + x*row + y] = d_data[stride + y*col + x];

        if(y < row && x < col) {
            sd_M[threadIdx.y][threadIdx.x] = d_data[stride + y*col + x];
        } else {
            sd_M[threadIdx.y][threadIdx.x] = 0.0f;
        }
        __syncthreads();

        if(y < row && x < col) {
            d_out[stride + x*row + y] = sd_M[threadIdx.y][threadIdx.x];
        }
        __syncthreads();

    }
}

__global__
void kScale(float *d_data, float factor, float offset, size_t N) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= N) return;

    d_data[tid] = d_data[tid] * factor + offset;
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


__global__
void kExp(float* d_data, float* d_out, int n_data) {
    int id = threadIdx.x + blockDim.x * blockIdx.x;

    if (id >= n_data) return;

    d_out[id] = expf(d_data[id]);
}

__global__
void kLog(float* d_data, float* d_out, int n_data) {
    int id = threadIdx.x + blockDim.x * blockIdx.x;

    if (id >= n_data) return;

    d_out[id] = logf(d_data[id]);
}




