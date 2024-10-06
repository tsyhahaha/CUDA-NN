#define BLOCK_SIZE 16
#define TILE_SIZE 4
#define TIED_SIZE 2

#ifndef KERNELS_H
#define KERNELS_H


__global__ void deviceMatmul_1D(
    float *d_A, float *d_B, float *d_C, int M, int N, int K
);
__global__ void deviceMatmul_2D(
    float *d_A, float *d_B, float *d_C, int M, int N, int K
);
__global__ void deviceMatmul_2D_shared(
    float *d_A, float *d_B, float *d_C, int M, int N, int K
);

__global__ void deviceMatmul_2D_register(
    float *d_A, float *d_B, float *d_C, int M, int N, int K
);

__global__ void deviceMatmul_2D_shared_register_1st(
    float *d_A, float *d_B, float *d_C, int M, int N, int K
);

__global__ void deviceMatmul_2D_shared_register_2nd(
    float *d_A, float *d_B, float *d_C, int M, int N, int K
);

__global__ void deviceMatmul_2D_shared_register_float4_1st(
    float *d_A, float *d_B, float *d_C, int M, int N, int K
);

#endif
