#include "../common.cuh"
#include "configure.cuh"

#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])

/*
assumption:
 - TILE_SIZE % 4 == 0
 - N % 4 == 0
 - TM/TN % 4 == 0
*/
__global__ void matmul_v3(
    float *d_A, float *d_B, float *d_C, int M, int K, int N
) {
    int row = (blockIdx.y * blockDim.y + threadIdx.y)*TM;
    int col = (blockIdx.x * blockDim.x + threadIdx.x)*TN;

    if(row >= M || col >= N) return;

    __shared__ float ds_A[TILE_SIZE][BLOCK_SIZE * TM];
    __shared__ float ds_B[TILE_SIZE][BLOCK_SIZE * TN];

    float tie[TM][TN] = {0.0f};
    float reg_A[TM] = {0.0f};
    float reg_B[TN] = {0.0f};
    float tmp[4] = {0.0f};

    int phase = (K - 1) / TILE_SIZE + 1;
    for(int p=0; p<phase;p++) {
        for(int i=0; i<TM; i++) {
            if(threadIdx.x*4 < TILE_SIZE) {
                if(row+i < M && p*TILE_SIZE + threadIdx.x*4 < K) {
                    FETCH_FLOAT4(tmp[0]) = FETCH_FLOAT4(d_A[(row+i) * K + p*TILE_SIZE + threadIdx.x*4]);

                    ds_A[threadIdx.x*4+0][threadIdx.y*TM+i] = tmp[0];
                    ds_A[threadIdx.x*4+1][threadIdx.y*TM+i] = tmp[1];
                    ds_A[threadIdx.x*4+2][threadIdx.y*TM+i] = tmp[2];
                    ds_A[threadIdx.x*4+3][threadIdx.y*TM+i] = tmp[3];
                } else {
                    ds_A[threadIdx.x*4+0][threadIdx.y*TM+i] = 0.0f;
                    ds_A[threadIdx.x*4+1][threadIdx.y*TM+i] = 0.0f;
                    ds_A[threadIdx.x*4+2][threadIdx.y*TM+i] = 0.0f;
                    ds_A[threadIdx.x*4+3][threadIdx.y*TM+i] = 0.0f;

                }
            }
        }
        

        for(int j=0; j<TN/4; j++) {
            if(threadIdx.y < TILE_SIZE) {
                if (col + j*4 < N && p*TILE_SIZE + threadIdx.y < K) {
                    FETCH_FLOAT4(ds_B[threadIdx.y][threadIdx.x*TN + j*4]) = FETCH_FLOAT4(d_B[(p*TILE_SIZE + threadIdx.y)*N + col + j*4]);
                } else {
                    ds_B[threadIdx.y][threadIdx.x*TN + j*4 + 0] = 0.0f;
                    ds_B[threadIdx.y][threadIdx.x*TN + j*4 + 1] = 0.0f;
                    ds_B[threadIdx.y][threadIdx.x*TN + j*4 + 2] = 0.0f;
                    ds_B[threadIdx.y][threadIdx.x*TN + j*4 + 3] = 0.0f;
                }
            }
        }
        __syncthreads();

        for (int k=0; k<TILE_SIZE; k++) {
            for (int i=0; i<TM/4; i++) {
                if(row + i*4 < M) {
                    FETCH_FLOAT4(reg_A[i*4]) = FETCH_FLOAT4(ds_A[k][threadIdx.y * TM + i*4]);
                }
            }

            for(int j=0; j<TN/4; j++) {
                if(col + j*4 < N) {
                    FETCH_FLOAT4(reg_B[j*4]) = FETCH_FLOAT4(ds_B[k][threadIdx.x * TN + j*4]);
                }
            }
            __syncthreads();

            for(int j=0; j<TN; j++) {
                for (int i=0; i<TM; i++) {
                    tie[i][j] += reg_A[i] * reg_B[j];
                }
            }
        }
    }

    for(int i=0; i<TM; i++) {
        for(int j=0; j<TN/4; j++) {
            if (row + i < M && col + j*4 < N) {
                FETCH_FLOAT4(d_C[(row + i) * N + col + j*4]) = FETCH_FLOAT4(tie[i][j*4]);
            }
        }
    }
}

int main(){
    // select which kernel to test
    KernelMatmul kernel = matmul_v3;
    const char* name = FUNC_NAME(matmul_v3);

    size_t M = 1024, K = 1024, N = 1024;
    size_t nBytes_A = M * K * sizeof(float);
    size_t nBytes_B = K * N * sizeof(float);
    size_t nBytes_C = M * N * sizeof(float);

    // alloc host memory
    float *h_A, *h_B, *h_C, *h_CC;
    const char* path_A = DATA_PATH;
    const char* path_B = DATA_PATH;
    h_A = loadWeightsFromTxt(path_A, {M, K});
    h_B = loadWeightsFromTxt(path_B, {K, N});
    h_C = (float *)malloc(nBytes_C);
    h_CC = (float *)malloc(nBytes_C); // saved for output of cpu matmul

    // alloc device memory
    float *d_A, *d_B, *d_C;
    CHECK(cudaMalloc((float **)&d_A, nBytes_A));
    CHECK(cudaMalloc((float **)&d_B, nBytes_B));
    CHECK(cudaMalloc((float **)&d_C, nBytes_C));
    // mv data from Host to Device
    CHECK(cudaMemcpy(d_A, h_A, nBytes_A, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_B, nBytes_B, cudaMemcpyHostToDevice));

    // kernel launch config
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    int row = (M-1)/(BLOCK_SIZE*TN) + 1, col = (N-1)/(BLOCK_SIZE*TM) + 1;
    dim3 grid(col, row);

    // warm up
    kernel<<<grid, block>>>(d_A, d_B, d_C, M, K, N); CHECK_KERNEL();

    // timer
    cudaEvent_t start, stop;
    float ker_time = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    for (int i=0; i<REPEAT_TIMES; i++) {
        kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ker_time, start, stop);
    printf("Time elapsed on `%s` of %ldx%ld @ %ldx%ld on GPU: %f ms.\n\n", name, M, K, K, N, ker_time/REPEAT_TIMES);

    // get the output of GPU
    CHECK(cudaMemcpy(h_C, d_C, nBytes_C, cudaMemcpyDeviceToHost));
    CHECK(cudaDeviceSynchronize());

    /////////////////////////////////////
    // check the output, CPU vs GPU
    h_CC = loadWeightsFromTxt("./host_result.txt", {M*N});
    check_result(h_C, h_CC, M*N);
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

