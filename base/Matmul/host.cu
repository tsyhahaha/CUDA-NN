#include "../common.cuh"
#include "configure.cuh"

void hostMatmul(
    float *h_a, float *h_b, float *h_result, int m, int n, int k
) {
    float tmp = 0.0f;
    for (int i = 0; i < m; ++i) 
    {
        for (int j = 0; j < k; ++j) 
        {
            for (int h = 0; h < n; ++h) 
            {
                tmp += h_a[i * n + h] * h_b[h * k + j];
            }
            h_result[i * k + j] = tmp;
            tmp = 0.0f;
        }
    }
}

int main(){
    size_t M = 1024, K = 1024, N = 1024;
    size_t nBytes_C = M * N * sizeof(float);

    // alloc host memory
    float *h_A, *h_B, *h_C;
    const char* path_A = "/home/tsyhahaha/CUDA-NN/data/feat.stn.fc3.weight.txt";
    const char* path_B = "/home/tsyhahaha/CUDA-NN/data/feat.stn.fc3.weight.txt";
    h_A = loadWeightsFromTxt(path_A, {M, K});
    h_B = loadWeightsFromTxt(path_B, {K, N});
    h_C = (float *)malloc(nBytes_C);

    hostMatmul(h_A, h_B, h_C, M, K, N);
    save_vector_to_txt("./host_result.txt", ptr_to_vec(h_C, M*N));

    // free memory
    free(h_A);
    free(h_B);
    free(h_C);
    return 0;
}

