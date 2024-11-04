#include "../common.cuh"
#include "configure.cuh"

#include <cuda.h>
#include <cublas_v2.h>

#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])

int main(){
    // nvcc xxx.cu -o xxx -lcublas -lcudart

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

    cublasHandle_t handle; // cublas句柄
    cublasCreate(&handle); // 初始化句柄
    float alpha = 1.0;
    float beta = 0.0;
    // cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, dB, N, dA, K, &beta, dC, N);
    // cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, M, N, K, &alpha, dA, K, dB, N, &beta, dC, M);
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                 N, M, K,
                 &alpha,
                 d_B, CUDA_R_32F, N,
                 d_A, CUDA_R_32F, K,
                 &beta,
                 d_C, CUDA_R_32F, N,
                 CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    // timer
    cudaEvent_t start, stop;
    float ker_time = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    for (int i=0; i<REPEAT_TIMES; i++) {
        // cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, dB, N, dA, K, &beta, dC, N);
        // cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, M, N, K, &alpha, dA, K, dB, N, &beta, dC, M);
        cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                     N, M, K,
                     &alpha,
                     d_B, CUDA_R_32F, N,
                     d_A, CUDA_R_32F, K,
                     &beta,
                     d_C, CUDA_R_32F, N,
                     CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ker_time, start, stop);
    printf("Time elapsed on `cublas` of %ldx%ld @ %ldx%ld on GPU: %f ms.\n\n", M, K, K, N, ker_time/REPEAT_TIMES);

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

