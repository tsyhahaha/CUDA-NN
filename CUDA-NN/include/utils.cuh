#ifndef UTILS_H
#define UTILS_H

#include<stdio.h>
#include<iostream>
#include<vector>

#define DEBUG 2

#if defined(DEBUG) && DEBUG >= 1
 #define DEBUG_PRINT(fmt, args...) fprintf(stderr, "[DEBUG] %s(%d):%s(): " fmt, __FILE__, __LINE__, __func__, ##args)
#else
 #define DEBUG_PRINT(fmt, args...)
#endif

#define CHECK(call)                                   \
do                                                    \
{                                                     \
    const cudaError_t error_code = call;              \
    if (error_code != cudaSuccess)                    \
    {                                                 \
        printf("CUDA error: \ncode=%d, name=%s, description=%s\n%s(%d)\n", error_code, cudaGetErrorName(error_code), cudaGetErrorString(error_code),  __FILE__, __LINE__); \
        exit(1);                                      \
    }                                                 \
} while (0)

#define ERROR(fmt, args...) do { fprintf(stderr, "%s(%d):%s(): " fmt, __FILE__, __LINE__, __func__, ##args); exit(0);} while(0)


#define CHECK_KERNEL() \
do \
{  \
    CHECK(cudaGetLastError());       \
    CHECK(cudaDeviceSynchronize());  \
} while(0)

void setGPU(const int GPU_idx);

float* loadWeightsFromTxt(const char* filename, std::vector<size_t> shape);

void printShape(std::vector<size_t> shape);

void printM(float* weight, const std::vector<size_t> shape);

float randomFloat(float a, float b);

int randomInt(int a, int b);
#endif /* !UTILS_H */
   