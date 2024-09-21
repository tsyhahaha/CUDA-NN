#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include "kernels.cuh"
#include "../common.cuh"

#define FUNC_NAME(f) #f

#define TEST_KERNEL(f)                                            \
do {                                                              \
    for (int i=0; i<= compute_times; i++) {                       \
        if (i==0) cudaEventRecord(start, 0);                      \
        f<<<grid_dim, block_dim>>>(d_A, d_B, d_C, M, N, K);       \
        CHECK(cudaGetLastError());                                \
        CHECK(cudaDeviceSynchronize());                           \
    }                                                             \
    CHECK(cudaMemcpy(h_C, d_C, nBytes_C, cudaMemcpyDeviceToHost));\
    CHECK(cudaMemcpy(h_activations, d_activations, nBytes_C, cudaMemcpyDeviceToHost)); \
    CHECK(cudaDeviceSynchronize());                               \
    cudaEventRecord(stop, 0);                                     \
    cudaEventSynchronize(stop);                                   \
    cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);      \
    is_right = check_result(h_CC, h_C, M, K);                     \
    printf("Time elapsed on `%s` of %dx%d . %dx%d on GPU: %f ms.\n\n", FUNC_NAME(f), M, N, N, K, gpu_elapsed_time_ms/compute_times); \
    if(!is_right and M<=16 and K<=16) printM(h_C, M, K);                              \
} while(0)

/*
***************************************************************
Matrix multiplication on CPU
***************************************************************
*/
void hostMatmul(
    float *h_a, float *h_b, float *h_result, int m, int n, int k
) {
    float tmp = 0.0f;
    for (int i = 0; i < m; ++i) 
    {
        for (int j = 0; j < k; ++j) 
        {
            for (int h = 0; h < n; ++h) 
            {
                tmp += h_a[i * n + h] * h_b[h * k + j];
            }
            h_result[i * k + j] = tmp;
            tmp = 0.0f;
        }
    }
}

int check_result(
    float *right, float *test, int m, int k
) {
    bool is_right = 1;
    float error_sum = 0.0f, l1_error;
    for (int i=0; i<m; i++) {
        for (int j=0; j<k; j++) {
            if (abs(right[i*k + j] - test[i*k + j]) > 1e-4) {
                l1_error = right[i*k + j] - test[i*k + j];
                error_sum += l1_error * l1_error;
                is_right = 0;
            }
        }
    }
    if (is_right) {
        printf("[AC] ");
    } else {
        printf("[MEAN ERROR]: %f\n", error_sum / (m*k));
        printf("[WA] ");
    }
    return is_right;
}

int main(void) {
    setGPU(0);

    // definition
    bool is_right;
    int compute_times = 10;
    float gpu_elapsed_time_ms, cpu_elapsed_time_ms;
    int M = 16, N = 32, K = 32;
    size_t nBytes_A = M * N * sizeof(float);
    size_t nBytes_B = N * K * sizeof(float);
    size_t nBytes_C = M * K * sizeof(float);
    
    // alloc host memory
    float *h_A, *h_B, *h_C, *h_activations, *h_CC;
    const char* path_A = "/home/taosiyuan/cudaCode/CUDA-programming/data/bn1.weight.txt";
    // const char* path_B = "/home/taosiyuan/cudaCode/CUDA-programming/data/bn1.weight.txt";
    const char* path_B = "/home/taosiyuan/cudaCode/CUDA-programming/data/bn2.weight.txt";
    h_A = loadWeights(path_A, M, N);
    h_B = loadWeights(path_B, N, K);
    h_C = (float *)malloc(nBytes_C);
    h_CC = (float *)malloc(nBytes_C);
    h_activations = (float *)malloc(nBytes_C);

    // some events to count the execution time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // start the CPU version
    for (int i=0; i<= compute_times; i++) {
        if (i==0) cudaEventRecord(start, 0);
        hostMatmul(h_A, h_B, h_CC, M, N, K);
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cpu_elapsed_time_ms, start, stop);
    printf("Time elapsed on `hostMatmul` of %dx%d . %dx%d on CPU: %f ms.\n\n", M, N, N, K, cpu_elapsed_time_ms/compute_times);
    if(M <= 16 && K <= 16) printM(h_CC, M, K);

    // alloc device memory
    float *d_A, *d_B, *d_C, *d_activations;
    CHECK(cudaMalloc((float **)&d_A, nBytes_A));
    CHECK(cudaMalloc((float **)&d_B, nBytes_B));
    CHECK(cudaMalloc((float **)&d_C, nBytes_C));
    CHECK(cudaMalloc((float **)&d_activations, nBytes_C));

    // mv data from Host to Device
    CHECK(cudaMemcpy(d_A, h_A, nBytes_A, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_B, nBytes_B, cudaMemcpyHostToDevice));

    dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE);
    unsigned int grid_rows = (M - 1) / BLOCK_SIZE + 1;
    unsigned int grid_cols = (K - 1) / BLOCK_SIZE + 1;
    dim3 grid_dim(grid_cols, grid_rows);    // ATTENTION!

    ////////////////////////////////////////////////////////////
    // 2D block; 2D grid;
    TEST_KERNEL(deviceMatmul_2D);
    ////////////////////////////////////////////////////////////
    // 2D block; 2D grid; op1.1: use shared memory
    TEST_KERNEL(deviceMatmul_2D_shared);
    ////////////////////////////////////////////////////////
    // 2D block; 2D grid; op1.2: use register
    TEST_KERNEL(deviceMatmul_2D_register);
    ///////////////////////////////////////////////////////////  
    // 2D block; 2D grid; op2.1 use shared memory and register
    TEST_KERNEL(deviceMatmul_2D_shared_register_1st);
    TEST_KERNEL(deviceMatmul_2D_shared_register_2st);

    // free memory
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_CC);
    free(h_activations);
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_C));
    CHECK(cudaFree(d_activations));
    
    cudaDeviceReset();
    return 0;
}
