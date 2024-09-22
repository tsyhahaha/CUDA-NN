#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include "kernels.cuh"
#include "../common.cuh"

#define FUNC_NAME(f) #f

#define REDUCE2ONE 0

#define TEST_KERNEL(f)                                            \
do {                                                              \
    for (int i=0; i<= compute_times; i++) {                       \
        if (i==0) cudaEventRecord(start, 0);                      \
        f<<<grid_dim, block_dim>>>(d_M, d_out, N);                \
        int N_tmp = N, grid_tmp = grid_dim;                       \
        if (REDUCE2ONE)                                             \
            while(grid_tmp > 1) {                                     \
                N_tmp = (N_tmp - 1) / BLOCK_SIZE + 1;                 \
                grid_tmp = (N_tmp - 1) / BLOCK_SIZE + 1;              \
                f<<<grid_tmp, block_dim>>>(d_out, d_out, N_tmp);      \
            }                                                         \
        CHECK(cudaGetLastError());                                \
        CHECK(cudaDeviceSynchronize());                           \
    }                                                             \
    CHECK(cudaMemcpy(h_out, d_out, nBytes_out, cudaMemcpyDeviceToHost));\
    CHECK(cudaDeviceSynchronize());                               \
    cudaEventRecord(stop, 0);                                     \
    cudaEventSynchronize(stop);                                   \
    cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);      \
    is_right = check_result(hh_out, h_out, grid_dim);                     \
    printf("Time elapsed on `%s` of %dx1 on GPU: %f ms.\n\n", FUNC_NAME(f), N, gpu_elapsed_time_ms/compute_times); \
} while(0)

// global var
int block_num;

/*
***************************************************************
Matrix multiplication on CPU
***************************************************************
*/
void hostReduceSum(
    float *h_M, float *h_out, int l
) {
    if (REDUCE2ONE) {
        h_out[0] = 0;
        for (int i = 0; i < l; i++) 
        {
            h_out[0] += h_M[i];
        }
    } else {
        int block_num = (l-1)/BLOCK_SIZE + 1;
        for(int i=0; i<block_num; i++) {
            h_out[i] = 0;
            for(int j=0; j<BLOCK_SIZE; j++) {
                if (i*BLOCK_SIZE+j < l)
                    h_out[i] += h_M[i*BLOCK_SIZE+j];
            }
        }
    }

}

int check_result(
    float *right, float *test, int N
) {
    float sum_right = 0.0f, sum_test = 0.0f;
    bool is_right = 1;
    if (REDUCE2ONE) {
        is_right = abs(*right-*test) < 1e-4;
        if (is_right) {
            printf("[AC] ");
        } else {
            printf("[ER]: %f != %f(right)\n", *test, *right);
            printf("[WA] ");
        }
    } else {
        for(int i=0; i<block_num; i++) {
            sum_right += right[i];
        }
        for(int i=0; i<N; i++) {
            sum_test += test[i];
        }
        is_right = abs(sum_right - sum_test) < 1e-4 ? 1 : 0;

        if (is_right) {
            printf("[AC] ");
        } else {
            printf("[MEAN ERROR]: %f\n", abs(sum_right - sum_test));
            printf("[WA] ");
        }
    }
    
    return is_right;
}

int main(void) {
    setGPU(0);

    // definition
    bool is_right;
    int compute_times = 100;
    float gpu_elapsed_time_ms, cpu_elapsed_time_ms;
    int N = 512; // reduce vector
    size_t nBytes_M = N * sizeof(float);

    int block_dim = BLOCK_SIZE;
    block_num = (N - 1) / block_dim + 1;
    int grid_dim = block_num;
    size_t nBytes_out = block_num * sizeof(float);
    
    // alloc host memory
    float *h_M, *h_out, *hh_out;
    const char* path_M = "../../data/bn1.weight.txt";
    h_M = loadWeights(path_M, 1, N);
    h_out = (float *)malloc(nBytes_out);
    hh_out = (float *)malloc(block_num * sizeof(float));

    // some events to count the execution time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // start the CPU version
    for (int i=0; i<= compute_times; i++) {
        if (i==0) cudaEventRecord(start, 0);
        hostReduceSum(h_M, hh_out, N);
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cpu_elapsed_time_ms, start, stop);
    printf("Time elapsed on `hostReduceSum` of %dx1 on CPU: %f ms.\n\n", N, cpu_elapsed_time_ms/compute_times);
    printf("standard output: \n");
    if(REDUCE2ONE) {
        printf("%f\n\n", *hh_out);
    } else {
        printM(hh_out, 1, block_num);
    }

    // alloc device memory
    float *d_M, *d_out;
    CHECK(cudaMalloc((float **)&d_M, nBytes_M));
    CHECK(cudaMalloc((float **)&d_out, nBytes_out));
    CHECK(cudaMemcpy(d_M, h_M, nBytes_M, cudaMemcpyHostToDevice));

    ////////////////////////////////////////////////////////////
    TEST_KERNEL(reduceSum);
    ////////////////////////////////////////////////////////////
    TEST_KERNEL(reduceSum_op1);
    ////////////////////////////////////////////////////////////
    TEST_KERNEL(reduceSum_op2);
    ////////////////////////////////////////////////////////////
    int tmp = grid_dim;
    grid_dim = (grid_dim+1) / 2;
    TEST_KERNEL(reduceSum_op3);
    grid_dim = tmp;
    ////////////////////////////////////////////////////////////
    tmp = grid_dim;
    grid_dim = (grid_dim+1) / 2;
    TEST_KERNEL(reduceSum_op4);
    grid_dim = tmp;
    ////////////////////////////////////////////////////////////
    tmp = grid_dim;
    grid_dim = (grid_dim+1) / 2;
    TEST_KERNEL(reduceSum_op5);
    grid_dim = tmp;
    ///////////////////////////////////////////////////////////
    tmp = grid_dim;
    grid_dim = (grid_dim - 1) / NUM_PER_THREAD + 1;
    TEST_KERNEL(reduceSum_op6);
    grid_dim = tmp;


    // free memory
    free(h_M);
    free(h_out);
    free(hh_out);
    CHECK(cudaFree(d_M));
    CHECK(cudaFree(d_out));
    
    cudaDeviceReset();
    return 0;
}
