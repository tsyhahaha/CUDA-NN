#include<stdio.h>

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

void setGPU(const int GPU_idx){
    int count = 0;
    CHECK(cudaGetDeviceCount(&count));
    CHECK(cudaSetDevice(GPU_idx));
}

void CheckError(cudaError_t error, const char* filename, int line) {
    do {
        if(error != cudaSuccess) {
            printf("CUDA error: \ncode=%d, name=%s, description=%s\nfile=%s, line %d\n", error, cudaGetErrorName(error), cudaGetErrorString(error), filename, line);
        }
    } while(0);
    
}


float* loadWeights(
    const char* filename, int m, int n
) {
    int nBytes = m * n * sizeof(float);
    FILE* file = fopen(filename, "r");

    if (file == NULL) {
        perror("Error opening file");
        return NULL;
    }

    float *matrix = (float *) malloc(nBytes);

    if (matrix == NULL) {
        perror("Error allocating memory");
        fclose(file);
        return NULL;
    }

    for (int i=0; i<m; i++) {
        for (int j=0; j<n; j++) {
            if(fscanf(file, "%f", &matrix[i*n + j]) != 1) {
                free(matrix);
                fclose(file);
                return NULL;
            }
        }
    }
    fclose(file);
    return matrix;
}

void printM(float* weight, int m, int n) {
    if (weight == NULL) {
        printf("printM: Matrix is NULL\n");
        return;
    }
    for (int i=0; i<m; i++) {
        for (int j=0; j<n; j++) {
            printf("%.3f ", weight[i*n + j]);
        }
        printf("\n");
    }
    printf("\n");
}
