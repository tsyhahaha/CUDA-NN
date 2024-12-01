#include "kernels.cuh"

__global__
void kMaxBackprop(
    float* gradients, float* d_in, float* max_index, size_t N, size_t C, size_t L
) {
    // gradients(N, C), d_in(N, C, L), max_index(N, C)
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int z = threadIdx.z + blockIdx.z * blockDim.z;

    if(z >= N || y >= C || x >= L) return;

    int offset2d = z*C+y;
    int offset3d = offset2d * L + x;

    int m_idx = (int) max_index[offset2d];
    if(x == m_idx) {
        d_in[offset3d] = gradients[offset2d];
    } else {
        d_in[offset3d] = 0.0f;
    }
}

__global__ 
void kCheckNaN(float* data, int size, bool* result) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Ensure the thread is within matrix bounds
    if (idx < size) {
        if (isnan(data[idx])) {
            *result = true;  // Found NaN, set the result to true
        }
    }
}