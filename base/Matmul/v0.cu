#include "../common.cuh"
#include "configure.cuh"

__global__ void matmul(
    float *d_A, float *d_B, float *d_C, int M, int K, int N
) {
    float tmp = 0.0f;
    
    int row = threadIdx.x + blockIdx.x * blockDim.x;
    int col = threadIdx.y + blockIdx.y * blockDim.y;
    if (row >= M || col >= N) return;

    for (int i = 0; i < K; i++){
        tmp += d_A[row * K + i] * d_B[i * N + col];
    }
    d_C[row * N + col] = tmp;
}

__global__ void matmul_v0(
    float *d_A, float *d_B, float *d_C, int M, int K, int N
) {
    float tmp = 0.0f;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row >= M || col >= N) return;

    for (int i=0; i<N; i++) {
        tmp += d_A[row*K + i] * d_B[i*N + col];
    }

    d_C[row*N + col] = tmp;
}

int main(){
    // select which kernel to test
    KernelMatmul kernel = matmul_v0;
    const char* name = FUNC_NAME(matmul_v0);

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
    dim3 grid(row, col);    // matmul
    // dim3 grid(col, row);    // matmul_v0

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

