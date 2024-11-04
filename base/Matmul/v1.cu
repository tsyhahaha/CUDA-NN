#include "../common.cuh"
#include "configure.cuh"

__global__ void matmul_v1(
    float *d_A, float *d_B, float *d_C, int M, int K, int N
) {
    float cVal = 0.0f;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float ds_A[BLOCK_SIZE][TILE_SIZE];
    __shared__ float ds_B[TILE_SIZE][BLOCK_SIZE];
    int phase = (K - 1) / TILE_SIZE + 1;
    for(int p=0; p<phase;p++) {
        if(threadIdx.x < TILE_SIZE) {
            if(row < M && p*TILE_SIZE + threadIdx.x < K) {
                ds_A[threadIdx.y][threadIdx.x] = d_A[row * N + p*TILE_SIZE + threadIdx.x];
            } else {
                ds_A[threadIdx.y][threadIdx.x] = 0.0f;
            }
        }

        if(threadIdx.y < TILE_SIZE) {
            if (col < N && p*TILE_SIZE + threadIdx.y < K) {
                ds_B[threadIdx.y][threadIdx.x] = d_B[(p*TILE_SIZE + threadIdx.y)*K + col];
            } else {
                ds_B[threadIdx.y][threadIdx.x] = 0.0f;
            }
        }
        __syncthreads();

        for (int i=0; i<TILE_SIZE; i++) {
            cVal += ds_A[threadIdx.y][i] * ds_B[i][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < K)
        d_C[row*K + col] = cVal;
}

int main(){
    // select which kernel to test
    KernelMatmul kernel = matmul_v1;
    const char* name = FUNC_NAME(matmul_v1);

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
    int row = (M-1)/BLOCK_SIZE + 1, col = (N-1)/BLOCK_SIZE + 1;
    dim3 grid(col, row);    // matmul_v1

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
    // TODO: check the output, CPU vs GPU
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

