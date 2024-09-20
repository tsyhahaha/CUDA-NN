#include<stdio.h>
#include<stdlib.h>

#include "kernels.h"

int main(void) {
    setGPU(0);

    // definition
    int M = 1, N = 512, K = 1;
    size_t nBytes_A = M * N * sizeof(float);
    size_t nBytes_B = N * K * sizeof(float);
    size_t nBytes_C = M * K * sizeof(float);
    
    // alloc host memory
    float *h_A, *h_B, *h_C, *h_activations;
    const char* path_A = "/home/taosiyuan/cudaCode/CUDA-programming/data/bn1.weight.txt";
    const char* path_B = "/home/taosiyuan/cudaCode/CUDA-programming/data/bn1.weight.txt";
    h_A = loadWeights(path_A, M, N);
    h_B = loadWeights(path_B, N, K);
    h_C = (float *)malloc(nBytes_C);
    h_activations = (float *)malloc(nBytes_C);

    // alloc device memory
    float *d_A, *d_B, *d_C, *d_activations;
    CHECK(cudaMalloc((float **)&d_A, nBytes_A));
    CHECK(cudaMalloc((float **)&d_B, nBytes_B));
    CHECK(cudaMalloc((float **)&d_C, nBytes_C));
    CHECK(cudaMalloc((float **)&d_activations, nBytes_C));

    // mv data from Host to Device
    CHECK(cudaMemcpy(d_A, h_A, nBytes_A, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_B, nBytes_B, cudaMemcpyHostToDevice));

    // run the kernel func
    /////////////////////////////////////////
    // 1D block; 1D grid;
    // int block_dim = 128; 
    // dim3 grid_dim((M*N-1)/128+1);
    // deviceMatmul_1D<<<grid_dim, block_dim>>>(d_A, d_B, d_C, M, N, K);
    /////////////////////////////////////////
    // 2D block; 2D grid;
    const int BLOCK_SIZE = 8;
    dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE); // 2D block
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 grid_dim(grid_rows, grid_cols);
    deviceMatmul_2D<<<grid_dim, block_dim>>>(d_A, d_B, d_C, M, N, K);
    /////////////////////////////////////////

    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    // mv result to Host
    CHECK(cudaMemcpy(h_C, d_C, nBytes_C, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_activations, d_activations, nBytes_C, cudaMemcpyDeviceToHost));

    // print and check
    printf("Matrix C:\n");
    printM(h_C, M, K);
    printf("\nActivations of C:\n");
    printM(h_activations, M, K);

    // free memory
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_activations);
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_C));
    CHECK(cudaFree(d_activations));
    
    cudaDeviceReset();
    return 0;
}