#include "utils.cuh"

void setGPU(const int GPU_idx){
    int count = 0;
    CHECK(cudaGetDeviceCount(&count));
    CHECK(cudaSetDevice(GPU_idx));
}

float* loadWeightsFromTxt(const char* filename, std::vector<size_t> shape) {
    FILE* file = fopen(filename, "r");
    if (file == NULL) {
        perror("Error opening file");
        return NULL;
    }

    int dim = shape.size();
    size_t n_data = 1;
    for(size_t s: shape) {
        n_data *= s;
    }
    int nBytes = n_data * sizeof(float);

    float *matrix = (float *) malloc(nBytes);

    if (matrix == NULL) {
        perror("Error allocating memory");
        fclose(file);
        return NULL;
    }


    if(dim == 1) {
        for (int i=0; i<shape[0]; i++) {
            if(fscanf(file, "%f", &matrix[i]) != 1) {
                free(matrix);
                fclose(file);
                return NULL;
            }
        }
    }else if(dim == 2) {
        for (int i=0; i<shape[0]; i++) {
            for (int j=0; j<shape[1]; j++) {
                if(fscanf(file, "%f", &matrix[i*shape[1] + j]) != 1) {
                    free(matrix);
                    fclose(file);
                    return NULL;
                }
            }
        }
    } else if(dim == 3) {
        for (int i=0; i<shape[0]; i++) {
            for (int j=0; j<shape[1]; j++) {
                for(int k=0; k<shape[2]; k++) {
                    if(fscanf(file, "%f", &matrix[i*shape[1]*shape[2] + j*shape[2] + k]) != 1) {
                        free(matrix);
                        fclose(file);
                        return NULL;
                    }
                }
            }
        }
    } else {
        ERROR("Dimension Error!");
    }

    fclose(file);
    return matrix;
}

void printM(float* weight, const std::vector<size_t> shape) {
    printShape(shape);
    int dim = shape.size();
    if (weight == NULL) {
        printf("print_M: Matrix is NULL\n");
        return;
    }
    if (dim == 1) {
        int m = shape[0];
        for(int i=0; i<m; i++) {
            printf("%.5f ", weight[i]);
        }
        printf("\n");
    } else if (dim == 2) {
        int m = shape[0], n = shape[1];
        for (int i=0; i<m; i++) {
            for (int j=0; j<n; j++) {
                printf("%.5f ", weight[i*n + j]);
            }
            printf("\n");
        }
        printf("\n");
    } else if(dim == 3) {
        int bz = shape[0], m = shape[1], n = shape[2];
        for(int b = 0; b < bz; b++) {
            printf("--------------batch %d---------------\n", b+1);
            for (int i=0; i<m; i++) {
                printf("[");
                for (int j=0; j<n; j++) {
                    printf("%.5f ", weight[b * m * n + i * n + j]);
                }
                printf("]\n");
            }
        }
    } else {
        printf("print_M: dim > 3 not implemented!");
        exit(0);
    }
}


void save_vector_to_txt(std::string& filename, std::vector<float>& data) {
    std::ofstream file(filename);
    if (file) {
        for (const float& value : data) {
            file << value << "\n";
        }
        file.close();
        DEBUG_PRINT("Save %s\n", filename.c_str());
    } else {
        ERROR("Unable to open file: %s\n", filename.c_str());
    }
}


void randomFloatMatrix(float* data, int len, float min, float max) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(min, max);
    for(int i=0; i<len; i++) {
        data[i] = dist(gen);
    }
}

float randomFloat(float a, float b) {
    return a + static_cast <float> (std::rand()) /( static_cast <float> (RAND_MAX/(b-a)));
}

int randomInt(int a, int b) {
    return a + std::rand()%(b-a);
}

unsigned long long generateRandomSeed() {
    std::random_device rd;  
    std::mt19937_64 gen(rd()); 
    return gen();
}

