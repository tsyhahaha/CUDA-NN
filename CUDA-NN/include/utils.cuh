#ifndef UTILS_H
#define UTILS_H

#include<stdio.h>
#include<iostream>
#include<vector>

#define CHECK(call)                                   \
do                                                    \
{                                                     \
    const cudaError_t error_code = call;              \
    if (error_code != cudaSuccess)                    \
    {                                                 \
        printf("CUDA error: \ncode=%d, name=%s, description=%s\nfile=%s, line %d\n", error_code, cudaGetErrorName(error_code), cudaGetErrorString(error_code),  __FILE__, __LINE__); \
        exit(1);                                      \
    }                                                 \
} while (0)


#define CHECK_KERNEL() \
do \
{  \
    CHECK(cudaGetLastError());       \
    CHECK(cudaDeviceSynchronize());  \
} while(0)

void setGPU(const int GPU_idx);

float* loadWeights(
    const char* filename, int m, int n
);

void print_M(float* weight, const std::vector<size_t>& shape);

#endif
   