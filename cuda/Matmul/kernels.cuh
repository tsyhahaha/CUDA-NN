#define BLOCK_SIZE 8
#define TILE_SIZE 8
#define TIED_SIZE 4

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

#endif
