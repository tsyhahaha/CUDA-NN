#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include "kernels.cuh"
#include "../common.cuh"

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
    for (int i=0; i<m; i++) {
        for (int j=0; j<k; j++) {
            if (abs(right[i*k + j] - test[i*k + j]) > 1e-4) {
                printf("%f != %f\n", right[i*k + j], test[i*k + j]);
                is_right = 0;
            }
        }
    }
    if (is_right) {
        printf("[AC] ");
    } else {
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
    int M = 64, N = 8, K = 64;
    size_t nBytes_A = M * N * sizeof(float);
    size_t nBytes_B = N * K * sizeof(float);
    size_t nBytes_C = M * K * sizeof(float);
    
    // alloc host memory
    float *h_A, *h_B, *h_C, *h_activations, *h_CC;
    const char* path_A = "/home/taosiyuan/cudaCode/CUDA-programming/data/bn1.weight.txt";
    const char* path_B = "/home/taosiyuan/cudaCode/CUDA-programming/data/bn1.weight.txt";
    h_A = loadWeights(path_A, M, N);
    h_B = loadWeights(path_B, N, K);
    h_C = (float *)malloc(nBytes_C);
    h_CC = (float *)malloc(nBytes_C);
    h_activations = (float *)malloc(nBytes_C);

    // some events to count the execution time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // start to count execution time of GPU version
    cudaEventRecord(start, 0);


    // start the CPU version
    for (int i=0; i<= compute_times; i++) {
        if (i==0) cudaEventRecord(start, 0);
        hostMatmul(h_A, h_B, h_CC, M, N, K);
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cpu_elapsed_time_ms, start, stop);
    printf("Time elapsed on `hostMatmul` of %dx%d . %dx%d on CPU: %f ms.\n\n", M, N, N, K, cpu_elapsed_time_ms/compute_times);

    if(M < 16 && K < 16) printM(h_CC, M, K);


    // alloc device memory
    float *d_A, *d_B, *d_C, *d_activations;
    CHECK(cudaMalloc((float **)&d_A, nBytes_A));
    CHECK(cudaMalloc((float **)&d_B, nBytes_B));
    CHECK(cudaMalloc((float **)&d_C, nBytes_C));
    CHECK(cudaMalloc((float **)&d_activations, nBytes_C));

    // mv data from Host to Device
    CHECK(cudaMemcpy(d_A, h_A, nBytes_A, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_B, nBytes_B, cudaMemcpyHostToDevice));

    // run the 1D kernel func
    ///////////////////////////////////////////////////////////
    // 1D block; 1D grid;
    // int block_dim = 128; 
    // dim3 grid_dim((M*K-1)/block_dim + 1);
    // for (int i=0; i<= compute_times; i++) {
    //     if (i==0) cudaEventRecord(start, 0);
    //     deviceMatmul_1D<<<grid_dim, block_dim>>>(d_A, d_B, d_C, M, N, K);
    // }
    // CHECK(cudaMemcpy(h_C, d_C, nBytes_C, cudaMemcpyDeviceToHost));
    // CHECK(cudaMemcpy(h_activations, d_activations, nBytes_C, cudaMemcpyDeviceToHost));
    // CHECK(cudaDeviceSynchronize());

    // // time counting terminate
    // cudaEventRecord(stop, 0);
    // cudaEventSynchronize(stop);
    // cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);
    // printf("Time elapsed on `deviceMatmul_1D` of %dx%d . %dx%d on GPU: %f ms.\n\n", M, N, N, K, gpu_elapsed_time_ms/compute_times);


    ////////////////////////////////////////////////////////////
    // 2D block; 2D grid;
    dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE); // 2D block
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 grid_dim(grid_rows, grid_cols);
    for (int i=0; i<= compute_times; i++) {
        if (i==0) cudaEventRecord(start, 0);
        deviceMatmul_2D<<<grid_dim, block_dim>>>(d_A, d_B, d_C, M, N, K);
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());
    }
    CHECK(cudaMemcpy(h_C, d_C, nBytes_C, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_activations, d_activations, nBytes_C, cudaMemcpyDeviceToHost));
    CHECK(cudaDeviceSynchronize());

    // time counting terminate
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);
    is_right = check_result(h_CC, h_C, M, K);
    printf("Time elapsed on `deviceMatmul_2D` of %dx%d . %dx%d on GPU: %f ms.\n\n", M, N, N, K, gpu_elapsed_time_ms/compute_times);
    
    if(!is_right) printM(h_C, M, K);
    ////////////////////////////////////////////////////////////
    // 2D block; 2D grid; op1
    for (int i=0; i<= compute_times; i++) {
        if (i==0) cudaEventRecord(start, 0);
        deviceMatmul_2D_shared<<<grid_dim, block_dim>>>(d_A, d_B, d_C, M, N, K);
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());
    }
    CHECK(cudaMemcpy(h_C, d_C, nBytes_C, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_activations, d_activations, nBytes_C, cudaMemcpyDeviceToHost));
    CHECK(cudaDeviceSynchronize());

    // time counting terminate
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);
    is_right = check_result(h_CC, h_C, M, K);
    printf("Time elapsed on `deviceMatmul_2D_shared` of %dx%d . %dx%d on GPU: %f ms.\n\n", M, N, N, K, gpu_elapsed_time_ms/compute_times);

    if(!is_right) printM(h_C, M, K);
    ////////////////////////////////////////////////////////
    for (int i=0; i<= compute_times; i++) {
        if (i==0) cudaEventRecord(start, 0);
        deviceMatmul_2D_register<<<grid_dim, block_dim>>>(d_A, d_B, d_C, M, N, K);
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());
    }
    CHECK(cudaMemcpy(h_C, d_C, nBytes_C, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_activations, d_activations, nBytes_C, cudaMemcpyDeviceToHost));
    CHECK(cudaDeviceSynchronize());

    // time counting terminate
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);
    is_right = check_result(h_CC, h_C, M, K);
    printf("Time elapsed on `deviceMatmul_2D_register` of %dx%d . %dx%d on GPU: %f ms.\n\n", M, N, N, K, gpu_elapsed_time_ms/compute_times);

    if(!is_right) printM(h_C, M, K);
    ///////////////////////////////////////////////////////////  

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
