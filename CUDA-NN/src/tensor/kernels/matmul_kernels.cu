#include "kernels.cuh"

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
void kBatchMatmul3D(
    float*d_A, float* d_B, float*d_out, int B, int M, int N, int K
) {
    // A (B x M x N) @ B (B x N x K) = C (B x M x K)
    float cVal = 0.0f;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.z;

    if(b >= B || row >= M || col >= K) return;

    __shared__ float ds_A[BLOCK_SIZE2D][TILE_SIZE];
    __shared__ float ds_B[TILE_SIZE][BLOCK_SIZE2D];

    int offset1 = b* M * N;
    int offset2 = b * N * K;
    int offset3 = b * M * K;

    int phase = (N - 1) / TILE_SIZE + 1;
    for(int p=0; p<phase; p++) {
        if (threadIdx.x < TILE_SIZE) {
            if(row < M  && p*TILE_SIZE + threadIdx.x < N) {
                ds_A[threadIdx.y][threadIdx.x] = d_A[offset1 + row*N + p*TILE_SIZE + threadIdx.x];
            } else {
                ds_A[threadIdx.y][threadIdx.x] = 0.0f;
            }
        }

        if(threadIdx.y < TILE_SIZE) {
            if(col < K && p*TILE_SIZE + threadIdx.y < N) {
                ds_B[threadIdx.y][threadIdx.x] = d_B[offset2 + (p*TILE_SIZE + threadIdx.y)*K + col];
            } else {
                ds_B[threadIdx.y][threadIdx.x] = 0.0f;
            }
        }

        __syncthreads();
        for (int i=0; i<TILE_SIZE; i++) {
            // constant: ds_A's x , ds_B's y
            cVal += ds_A[threadIdx.y][i] * ds_B[i][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < K) {
        d_out[offset3 + row*K + col] = cVal;
    }
}

