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
        f<<<grid_dim, block_dim>>>(d_M, d_out, M, N);                \
        CHECK(cudaGetLastError());                                \
        CHECK(cudaDeviceSynchronize());                           \
    }                                                             \
    CHECK(cudaMemcpy(h_out, d_out, nBytes_out, cudaMemcpyDeviceToHost));\
    CHECK(cudaDeviceSynchronize());                               \
    cudaEventRecord(stop, 0);                                     \
    cudaEventSynchronize(stop);                                   \
    cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);      \
    is_right = check_result(hh_out, h_out, N, M);                     \
    printf("Time elapsed on `%s` of %dx1 on GPU: %f ms.\n\n", FUNC_NAME(f), N, gpu_elapsed_time_ms/compute_times); \
} while(0)

// global var
int block_num;

/*
***************************************************************
Matrix traspose on CPU
***************************************************************
*/
void hostTranspose(
    float *h_M, float *h_out, int m, int n
) {
    for(int i=0; i<m; i++) {
        for(int j=0; j<n; j++) {
            h_out[j*m + i] = h_M[i*n + j];
        }
    }
}

int check_result(
    float *right, float *test, int m, int k
) {
    bool is_right = 1;
    float error_sum = 0.0f;
    for (int i=0; i<m; i++) {
        for (int j=0; j<k; j++) {
            if (abs(right[i*k + j] - test[i*k + j]) > 1e-4) {
                error_sum = abs(right[i*k + j] - test[i*k + j]);
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
    int M = 16, N = 8; // reduce vector
    size_t nBytes_M = N * M * sizeof(float);

    dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE);
    unsigned int grid_rows = (M - 1) / BLOCK_SIZE + 1;
    unsigned int grid_cols = (N - 1) / BLOCK_SIZE + 1;
    dim3 grid_dim(grid_cols, grid_rows);    // ATTENTION!
    size_t nBytes_out = nBytes_M;
    
    // alloc host memory
    float *h_M, *h_out, *hh_out;
    const char* path_M = "../../data/bn1.weight.txt";
    h_M = loadWeights(path_M, M, N);
    h_out = (float *)malloc(nBytes_out);
    hh_out = (float *)malloc(nBytes_out);

    // some events to count the execution time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // start the CPU version
    for (int i=0; i<= compute_times; i++) {
        if (i==0) cudaEventRecord(start, 0);
        hostTranspose(h_M, hh_out, M, N);
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cpu_elapsed_time_ms, start, stop);
    printf("Time elapsed on `hostTranspose` of %dx1 on CPU: %f ms.\n\n", N, cpu_elapsed_time_ms/compute_times);
    
    if (M <= 16 && N <= 16){
        printf("standard output: \n");
        printM(hh_out, N, M);
    }
        

    // alloc device memory
    float *d_M, *d_out;
    CHECK(cudaMalloc((float **)&d_M, nBytes_M));
    CHECK(cudaMalloc((float **)&d_out, nBytes_out));
    CHECK(cudaMemcpy(d_M, h_M, nBytes_M, cudaMemcpyHostToDevice));

    ////////////////////////////////////////////////////////////
    TEST_KERNEL(transpose);
    ////////////////////////////////////////////////////////////
    TEST_KERNEL(transpose_shared);

    // free memory
    free(h_M);
    free(h_out);
    free(hh_out);
    CHECK(cudaFree(d_M));
    CHECK(cudaFree(d_out));
    
    cudaDeviceReset();
    return 0;
}
