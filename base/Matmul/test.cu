#include "../common.cuh"

#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])

#define BLOCK_SIZE3D 16
#define TILE_SIZE 16
#define BATCH_BASE 2

#define TM 4
#define TN 4

#define REPEAT_TIMES 5

__global__
void kConv1d_v1(float* d_in, float* d_out, float* weights, float* bias, int C_in, int C_out, int L, int N) {
    // weights(C_out x C_in) @ d_in(B x C_in x L) + bias(C_out)= (B x C_out x L)
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int batch = blockIdx.z * blockDim.z + threadIdx.z;

    __shared__ float ds_A[BLOCK_SIZE3D][TILE_SIZE];
    __shared__ float ds_B[BATCH_BASE][TILE_SIZE][BLOCK_SIZE3D];

    if(batch >= N) return;

    int offset_in  = batch * C_in  * L;
    int offset_out = batch * C_out * L;

    float cVal = 0.0f;
    int phase = (C_in - 1) / TILE_SIZE + 1;
    for(int p=0; p<phase;p++) {
        if (row < C_out && threadIdx.x < TILE_SIZE && threadIdx.z == 0) {
            if(p*TILE_SIZE + threadIdx.x < C_in) {
                ds_A[threadIdx.y][threadIdx.x] = weights[row*C_in + p*TILE_SIZE + threadIdx.x];
            } else {
                ds_A[threadIdx.y][threadIdx.x] = 0.0f;
            }
        }

        if(col < L && threadIdx.y < TILE_SIZE) {
            if(p*TILE_SIZE + threadIdx.y < C_in){
                ds_B[threadIdx.z][threadIdx.y][threadIdx.x] = d_in[offset_in + (p*TILE_SIZE + threadIdx.y)*L + col];
            } else {
                ds_B[threadIdx.z][threadIdx.y][threadIdx.x] = 0.0f;
            }
        }

        __syncthreads();
        for (int i=0; i<TILE_SIZE; i++) {
            // constant: ds_A's x , ds_B's y
            cVal += ds_A[threadIdx.y][i] * ds_B[threadIdx.z][i][threadIdx.x];
        }
        __syncthreads();
    }
    if(row < C_out && col < L)
        d_out[offset_out + row*L + col] = cVal + bias[row];
}

__global__
void kConv1d_v3(float* d_in, float* d_out, float* weights, float* bias, int C_in, int C_out, int L, int N) {
    // weights(C_out x C_in) @ d_in(B x C_in x L) + bias(C_out)= (B x C_out x L)
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int batch = blockIdx.z * blockDim.z + threadIdx.z;

    if(row * TM >= C_out || col * TN >= L || batch >= N) return;

    __shared__ float ds_A[TILE_SIZE][BLOCK_SIZE3D];
    __shared__ float ds_B[BATCH_BASE][TILE_SIZE][BLOCK_SIZE3D];

    float cVal[TM][TN] = {0.0f};
    float reg_A[TM] = {0.0f};
    float reg_B[TN] = {0.0f};

    float tmp[4]; // TODO

    int offset_in  = batch * C_in  * L;
    int offset_out = batch * C_out * L;

    int phase = (C_in - 1) / TILE_SIZE + 1;
    for(int p=0; p<phase;p++) {
        if (threadIdx.z == 0 && row < C_out && threadIdx.x * 4 < TILE_SIZE) {
            if(p*TILE_SIZE + threadIdx.x < C_in) {
                FETCH_FLOAT4(tmp[0]) = FETCH_FLOAT4(weights[row * C_in + p * TILE_SIZE + threadIdx.x * 4]);

                ds_A[threadIdx.x + 0][threadIdx.y] = tmp[0];
                ds_A[threadIdx.x + 1][threadIdx.y] = tmp[1];
                ds_A[threadIdx.x + 2][threadIdx.y] = tmp[2];
                ds_A[threadIdx.x + 3][threadIdx.y] = tmp[3];
            } else {
                ds_A[threadIdx.x + 0][threadIdx.y] = 0.0;
                ds_A[threadIdx.x + 1][threadIdx.y] = 0.0;
                ds_A[threadIdx.x + 2][threadIdx.y] = 0.0;
                ds_A[threadIdx.x + 3][threadIdx.y] = 0.0;
            }
        }

        if(col * 4 < L && threadIdx.y < TILE_SIZE) {
            if(p*TILE_SIZE + threadIdx.y < C_in){
                FETCH_FLOAT4(ds_B[threadIdx.z][threadIdx.y][threadIdx.x*4]) = FETCH_FLOAT4(d_in[offset_in + (p*TILE_SIZE + threadIdx.y)*L + col*4]);
            } else {
                ds_B[threadIdx.z][threadIdx.y][threadIdx.x*4 + 0] = 0.0f;
                ds_B[threadIdx.z][threadIdx.y][threadIdx.x*4 + 1] = 0.0f;
                ds_B[threadIdx.z][threadIdx.y][threadIdx.x*4 + 2] = 0.0f;
                ds_B[threadIdx.z][threadIdx.y][threadIdx.x*4 + 3] = 0.0f;
            }
        }
        __syncthreads();

        for (int k=0; k<TILE_SIZE; k++) {
            for (int i=0; i<TM/4; i++) {
                if(threadIdx.y * TM + i*4 < BLOCK_SIZE3D)
                    FETCH_FLOAT4(reg_A[i*4]) = FETCH_FLOAT4(ds_A[k][threadIdx.y * TM + i*4]);   // !!!!!!
            }

            for(int j=0; j<TN/4; j++) {
                if(threadIdx.x * TN + j*4 < BLOCK_SIZE3D)
                    FETCH_FLOAT4(reg_B[j*4]) = FETCH_FLOAT4(ds_B[threadIdx.z][k][threadIdx.x * TN + j*4]);
            }
            __syncthreads();


            for (int i=0; i<TM; i++) {
                for(int j=0; j<TN; j++) {
                    cVal[i][j] += reg_A[i] * reg_B[j];
                }
            }
        }
    }

    for(int i=0; i<TM; i++) {
        for(int j=0; j<TN; j++) {
            cVal[i][j] += bias[row * TM + i];
        }
    }

    for(int i=0; i<TM; i++) {
        for(int j=0; j<TN/4; j++) {
            if(row * TM + i < C_out && col * TN + j*4 < L)
                FETCH_FLOAT4(d_out[offset_out + (row * TM + i)*L + col * TN + j*4]) = FETCH_FLOAT4(cVal[i][j*4]);
        }
    }
    
}

__global__
void kConv1d_v2(float* d_in, float* d_out, float* weights, float* bias, int C_in, int C_out, int L, int N) {
    // weights(C_out x C_in) @ d_in(B x C_in x L) + bias(C_out)= (B x C_out x L)
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int batch = blockIdx.z * blockDim.z + threadIdx.z;

    __shared__ float ds_A[BLOCK_SIZE3D*TM][TILE_SIZE];
    __shared__ float ds_B[BATCH_BASE][TILE_SIZE][BLOCK_SIZE3D*TN];

    if(row >= C_out || col >= L || batch >= N) return;

    float cVal[TM][TN] = {0.0f};
    float reg_A[TM] = {0.0f};
    float reg_B[TN] = {0.0f};

    int offset_in  = batch * C_in  * L;
    int offset_out = batch * C_out * L;
    int stride_row = BLOCK_SIZE3D / TM;
    int stride_col = BLOCK_SIZE3D / TN;

    int phase = (C_in - 1) / TILE_SIZE + 1;
    for(int p=0; p<phase;p++) {
        if (threadIdx.z == 0 && row < C_out && threadIdx.x < TILE_SIZE) {
            if(p*TILE_SIZE + threadIdx.x < C_in) {
                ds_A[threadIdx.y][threadIdx.x] = weights[row*C_in + p*TILE_SIZE + threadIdx.x];
            } else {
                ds_A[threadIdx.y][threadIdx.x] = 0.0f;
            }
        }

        if(col < L && threadIdx.y < TILE_SIZE) {
            if(p*TILE_SIZE + threadIdx.y < C_in){
                ds_B[threadIdx.z][threadIdx.y][threadIdx.x] = d_in[offset_in + (p*TILE_SIZE + threadIdx.y)*L + col];
            } else {
                ds_B[threadIdx.z][threadIdx.y][threadIdx.x] = 0.0f;
            }
        }
        __syncthreads();
        
        #pragma unroll
        for (int k=0; k<TILE_SIZE; k++) {
            if(threadIdx.x < BLOCK_SIZE3D/TN && threadIdx.y < BLOCK_SIZE3D/TM)
            for (int i=0; i<TM; i++) {
                reg_A[i] = ds_A[threadIdx.y + i * stride_row][k];
            }
            if(threadIdx.x < BLOCK_SIZE3D/TN && threadIdx.y < BLOCK_SIZE3D/TM)
            for(int j=0; j<TN; j++) {
                reg_B[j] = ds_B[threadIdx.z][k][threadIdx.x + j * stride_col];
                
            }
            __syncthreads();
            if(threadIdx.x < BLOCK_SIZE3D/TN && threadIdx.y < BLOCK_SIZE3D/TM)
            for (int i=0; i<TM; i++) {
                for(int j=0; j<TN; j++) {
                    cVal[i][j] += reg_A[i] * reg_B[j];
                }
            }
        }
    }

    
    for(int i=0; i<TM; i++) {
        for(int j=0; j<TN; j++) {
            if(row + i * stride_row < C_out && col + j * stride_col < L && threadIdx.x < BLOCK_SIZE3D/TN && threadIdx.y < BLOCK_SIZE3D/TM) {
                d_out[offset_out + (row + i * stride_row)*L + col + j * stride_col] = cVal[i][j] + bias[row + i * stride_row];
            }
        }
    }
}

__global__
void kConv1d_v2_base(float* d_in, float* d_out, float* weights, float* bias, int C_in, int C_out, int L, int N) {
    // weights(C_out x C_in) @ d_in(B x C_in x L) + bias(C_out)= (B x C_out x L)
    int row = (blockIdx.y * blockDim.y + threadIdx.y)*TM;
    int col = (blockIdx.x * blockDim.x + threadIdx.x)*TN;
    int batch = blockIdx.z * blockDim.z + threadIdx.z;

    if(row >= C_out || col >= L || batch >= N) return;

    __shared__ float ds_w[TILE_SIZE][BLOCK_SIZE3D * TM];
    __shared__ float ds_in[BATCH_BASE][TILE_SIZE][BLOCK_SIZE3D * TN];

    float tie[TM][TN] = {0.0f};
    float reg_w[TM] = {0.0f};
    float reg_in[TN] = {0.0f};
    float tmp[4] = {0.0f};

    uint offset_in = batch * C_in * L;
    uint offset_out = batch * C_out * L;

    int phase = (C_in - 1) / TILE_SIZE + 1;
    for(int p=0; p<phase;p++) {
        for(int i=0; i<TM; i++) {
            if(threadIdx.z == 0 && threadIdx.x*4 < TILE_SIZE) {
                if(row+i < C_out && p*TILE_SIZE + threadIdx.x*4 < C_in) {
                    FETCH_FLOAT4(tmp[0]) = FETCH_FLOAT4(weights[(row+i) * C_in + p*TILE_SIZE + threadIdx.x*4]);

                    ds_w[threadIdx.x*4+0][threadIdx.y*TM+i] = tmp[0];
                    ds_w[threadIdx.x*4+1][threadIdx.y*TM+i] = tmp[1];
                    ds_w[threadIdx.x*4+2][threadIdx.y*TM+i] = tmp[2];
                    ds_w[threadIdx.x*4+3][threadIdx.y*TM+i] = tmp[3];

                    // ds_w[threadIdx.y*TM+i][threadIdx.x] = weights[(row+i) * C_in + p*TILE_SIZE + threadIdx.x];
                } else {
                    ds_w[threadIdx.x*4+0][threadIdx.y*TM+i] = 0.0f;
                    ds_w[threadIdx.x*4+1][threadIdx.y*TM+i] = 0.0f;
                    ds_w[threadIdx.x*4+2][threadIdx.y*TM+i] = 0.0f;
                    ds_w[threadIdx.x*4+3][threadIdx.y*TM+i] = 0.0f;

                    // ds_w[threadIdx.y*TM+i][threadIdx.x] = 0.0f;
                }
            }
        }
        

        for(int j=0; j<TN/4; j++) {
            if(threadIdx.y < TILE_SIZE) {
                if (col + j*4 < L && p*TILE_SIZE + threadIdx.y < C_in) {
                    FETCH_FLOAT4(ds_in[threadIdx.z][threadIdx.y][threadIdx.x*TN + j*4]) = FETCH_FLOAT4(d_in[offset_in + (p*TILE_SIZE + threadIdx.y)*N + col + j*4]);
                } else {
                    ds_in[threadIdx.z][threadIdx.y][threadIdx.x*TN + j*4 + 0] = 0.0f;
                    ds_in[threadIdx.z][threadIdx.y][threadIdx.x*TN + j*4 + 1] = 0.0f;
                    ds_in[threadIdx.z][threadIdx.y][threadIdx.x*TN + j*4 + 2] = 0.0f;
                    ds_in[threadIdx.z][threadIdx.y][threadIdx.x*TN + j*4 + 3] = 0.0f;

                    // ds_B[threadIdx.y][threadIdx.x*TN + j] = 0.0f;
                }
            }
        }
        __syncthreads();

        for (int k=0; k<TILE_SIZE; k++) {
            for (int i=0; i<TM/4; i++) {
                if(row + i*4 < C_out) {
                    FETCH_FLOAT4(reg_w[i*4]) = FETCH_FLOAT4(ds_w[k][threadIdx.y * TM + i*4]);
                }
            }

            for(int j=0; j<TN/4; j++) {
                if(col + j*4 < L) {
                    FETCH_FLOAT4(reg_in[j*4]) = FETCH_FLOAT4(ds_in[threadIdx.z][k][threadIdx.x * TN + j*4]);
                }
            }
            __syncthreads();

            for(int j=0; j<TN; j++) {
                for (int i=0; i<TM; i++) {
                    tie[i][j] += reg_w[i] * reg_in[j];
                }
            }
        }
    }

    for(int i=0; i<TM; i++) {
        for(int j=0; j<TN/4; j++) {
            if (row + i < C_out && col + j*4 < L) {
                FETCH_FLOAT4(d_out[offset_out + (row + i) * L + col + j*4]) = FETCH_FLOAT4(tie[i][j*4]);
            }
        }
    }
}

int main(){
    // select which kernel to test
    size_t C_out = 1024, C_in = 128, N = 4, L = 1024;
    size_t nBytes_A = C_out * C_in * sizeof(float);
    size_t nBytes_B = N * C_in * L * sizeof(float);
    size_t nBytes_C = N *  C_out * L * sizeof(float);
    size_t nBytes_bias = C_out * sizeof(float);

    // alloc host memory
    float *h_A, *h_B, *h_C, *h_bias, *h_CC;
    const char* path_A = "/home/tsyhahaha/CUDA-NN/data/feat.stn.fc3.weight.txt";
    const char* path_B = "/home/tsyhahaha/CUDA-NN/data/feat.stn.fc3.weight.txt";
    h_A = loadWeightsFromTxt(path_A, {C_out, C_in});
    h_B = loadWeightsFromTxt(path_B, {N, C_in, L});
    h_bias = loadWeightsFromTxt(path_B, {C_out});
    h_C = (float *)malloc(nBytes_C);
    h_CC = (float *)malloc(nBytes_C); // saved for output of cpu matmul

    // alloc device memory
    float *d_A, *d_B, *d_bias, *d_C;
    CHECK(cudaMalloc((float **)&d_A, nBytes_A));
    CHECK(cudaMalloc((float **)&d_B, nBytes_B));
    CHECK(cudaMalloc((float **)&d_C, nBytes_C));
    CHECK(cudaMalloc((float **)&d_bias, nBytes_bias));
    // mv data from Host to Device
    CHECK(cudaMemcpy(d_A, h_A, nBytes_A, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_B, nBytes_B, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_bias, h_bias, nBytes_bias, cudaMemcpyHostToDevice));

    // kernel launch config
    dim3 block(BLOCK_SIZE3D, BLOCK_SIZE3D, BATCH_BASE);
    int row = (L-1)/BLOCK_SIZE3D + 1, col = (C_out-1)/BLOCK_SIZE3D + 1;
    dim3 grid(row, col, (N-1)/BATCH_BASE + 1);

    // warm up
    kConv1d_v2_base<<<grid, block>>>(d_B, d_C, d_A,  d_bias, C_in, C_out, L, N); CHECK_KERNEL();

    // timer
    cudaEvent_t start, stop;
    float ker_time = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    for (int i=0; i<REPEAT_TIMES; i++) {
        kConv1d_v2_base<<<grid, block>>>(d_B, d_C, d_A, d_bias, C_in, C_out, L, N); CHECK_KERNEL();
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ker_time, start, stop);
    printf("Time elapsed on `conv1d` on GPU: %f ms.\n\n", ker_time/REPEAT_TIMES);

    // get the output of GPU
    CHECK(cudaMemcpy(h_C, d_C, nBytes_C, cudaMemcpyDeviceToHost));
    CHECK(cudaDeviceSynchronize());

    /////////////////////////////////////
    // TODO: check the output, CPU vs GPU
    // xxx
    /////////////////////////////////////

    // free memory
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_CC);
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_C));

    cudaDeviceReset();
    return 0;
}

