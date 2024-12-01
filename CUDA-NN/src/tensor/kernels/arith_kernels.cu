#include "kernels.cuh"

/* add and sub */

__global__
void kAdd_l1(
    float *d_A, float *d_B, float *d_out, int M, float f1, float f2
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col >= M) return;

    d_out[col] = f1 * d_A[col] + f2 * d_B[col];
}

__global__
void kAdd_l2(
    float *d_A, float *d_B, float *d_out, int M, int N, float f1, float f2
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    d_out[row * N + col] = f1 * d_A[row * N + col] + f2 * d_B[col];
}

__global__
void kAddStride_l1(
    float *d_A, float *d_B, float *d_out, int M, 
    float f1, float f2,
    int s1, int s2
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col >= M) return;

    int col_1 = col / s1;
    int col_2 = col / s2;

    d_out[col] = f1 * d_A[col_1] + f2 * d_B[col_2];
}



__global__
void kAddStride_l2(
    float *d_A, float *d_B, float *d_out, int M, int N, float f1, float f2, 
    int s11, int s12, int s21, int s22
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    int N1 = s12 == 1 ? N : 1;
    int N2 = s22 == 1 ? N : 1;

    d_out[row * N + col] = f1 * d_A[row / s11 * N1 + col / s12] + f2 * d_B[row / s21 * N2 + col / s22];
}

__global__
void kAddStride_l3(
    float *d_A, float *d_B, float *d_out, int B, int M, int N, float f1, float f2, 
    int s11, int s12, int s13, int s21, int s22, int s23
) {
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    int M1 = s12 == 1 ? M : 1;
    int N1 = s13 == 1 ? N : 1;
    int M2 = s22 == 1 ? M : 1;
    int N2 = s23 == 1 ? N : 1;

    if(z < B && y < M && x < N)
        d_out[z*(M*N) + y*N + x] = f1 * d_A[z/s11 * (M1 * N1) + y/s12 * N1 + x/s13] + f2 * d_B[z/s21*(M2 * N2) + y/s22 * N2 + x/s23];
}

/* dot mul */
__global__
void kDotStride_l1(
    float *d_A, float *d_B, float *d_out, int M, float f,
    int s1, int s2
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col >= M) return;

    int col_1 = col / s1;
    int col_2 = col / s2;

    d_out[col] = f * d_A[col_1] * d_B[col_2];
}

__global__
void kDotStride_l2(
    float *d_A, float *d_B, float *d_out, int M, int N, float f, 
    int s11, int s12, int s21, int s22
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    int N1 = s12 == 1 ? N : 1;
    int N2 = s22 == 1 ? N : 1;

    d_out[row * N + col] = f * d_A[row / s11 * N1 + col / s12] * d_B[row / s21 * N2 + col / s22];
}

__global__
void kDotStride_l3(
    float *d_A, float *d_B, float *d_out, int B, int M, int N, float f, 
    int s11, int s12, int s13, int s21, int s22, int s23
) {
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (z >= B || y >= M || x >= N) return;

    int N1 = s13 == 1 ? N : 1;
    int N2 = s23 == 1 ? N : 1;
    int M1 = s12 == 1 ? M : 1;
    int M2 = s22 == 1 ? M : 1;

    d_out[z*(M*N) + y*N + x] = f * d_A[z/s11 * (M1 * N1) + y/s12 * N1 + x/s13] * d_B[z/s21*(M2 * N2) + y/s22 * N2 + x/s23];
}

/* divide */

__global__
void kDivStride_l1(
    float *d_A, float *d_B, float *d_out, int M, float f,
    int s1, int s2
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col >= M) return;

    int col_1 = col / s1;
    int col_2 = col / s2;

    d_out[col] = f * d_A[col_1] / d_B[col_2];
}

__global__
void kDivStride_l2(
    float *d_A, float *d_B, float *d_out, int M, int N, float f, 
    int s11, int s12, int s21, int s22
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    int N1 = s12 == 1 ? N : 1;
    int N2 = s22 == 1 ? N : 1;

    d_out[row * N + col] = f * d_A[row / s11 * N1 + col / s12] / d_B[row / s21 * N2 + col / s22];
}

__global__
void kDivStride_l3(
    float *d_A, float *d_B, float *d_out, int B, int M, int N, float f, 
    int s11, int s12, int s13, int s21, int s22, int s23
) {
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (z >= B || y >= M || x >= N) return;

    int N1 = s13 == 1 ? N : 1;
    int N2 = s23 == 1 ? N : 1;
    int M1 = s12 == 1 ? M : 1;
    int M2 = s22 == 1 ? M : 1;

    d_out[z*(M*N) + y*N + x] = f * d_A[z/s11 * (M1 * N1) + y/s12 * N1 + x/s13] / d_B[z/s21*(M2 * N2) + y/s22 * N2 + x/s23];
}
