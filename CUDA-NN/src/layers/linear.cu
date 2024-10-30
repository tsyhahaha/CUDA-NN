#include "linear.cuh"

__global__
void kLinear2D(float* input, float* d_out, float* weights, float* bias, int M, int N, int K) {
    // input(N x in_features) @ weights(out_features x in_features).T + bias(out_features)
    float cVal = 0.0f;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float ds_A[BLOCK_SIZE2D][TILE_SIZE];
    __shared__ float ds_B[BLOCK_SIZE2D][TILE_SIZE];

    int phase = (N - 1) / TILE_SIZE + 1;
    for(int p=0; p<phase;p++) {
        if (row < M && threadIdx.x < TILE_SIZE) {
            if(p*TILE_SIZE + threadIdx.x < N) {
                ds_A[threadIdx.y][threadIdx.x] = input[row*N + p*TILE_SIZE + threadIdx.x];
            } else {
                ds_A[threadIdx.y][threadIdx.x] = 0.0f;
            }
       	}
        
        if (col < K && threadIdx.y < TILE_SIZE) {
            if(p*TILE_SIZE+threadIdx.y < N) {
                ds_B[threadIdx.x][threadIdx.y] = weights[col*N + p*TILE_SIZE + threadIdx.y];
            } else {
                ds_B[threadIdx.x][threadIdx.y] = 0.0f;
            }

        }

        __syncthreads();
        for (int i=0; i<TILE_SIZE; i++) {
            // constant: ds_A's x , ds_B's x
            cVal += ds_A[threadIdx.y][i] * ds_B[threadIdx.x][i];
        }
        __syncthreads();
    }

    if(row < M && col < K)
        d_out[row*K + col] = cVal + bias[col];
}

__global__
void kLinear2D_v1(float* input, float* d_out, float* weights, float* bias, int M, int K, int N) {
    // input(N x in_features) @ weights(out_features x in_features).T + bias(out_features)
    int row = (blockIdx.y * blockDim.y + threadIdx.y)*TM;
    int col = (blockIdx.x * blockDim.x + threadIdx.x)*TN;

    if(row >= M || col >= N) return;

    __shared__ float ds_A[TILE_SIZE][BLOCK_SIZE2D * TM];
    __shared__ float ds_B[TILE_SIZE][BLOCK_SIZE2D * TN];

    float tie[TM][TN] = {0.0f};
    float reg_A[TM] = {0.0f};
    float reg_B[TN] = {0.0f};

    int phase = (K - 1) / TILE_SIZE + 1;
    for(int p=0; p<phase;p++) {
        for(int i=0; i<TM; i++) {
            if(threadIdx.x < TILE_SIZE) {
                if(row + i < M && p*TILE_SIZE + threadIdx.x < K) {
                    ds_A[threadIdx.x][threadIdx.y*TM + i] = input[(row + i) * K + p*TILE_SIZE + threadIdx.x];
                } else {
                    ds_A[threadIdx.x][threadIdx.y*TM + i] = 0.0f;
                }
            }
        }
        
        for(int j=0; j<TN; j++) {
            if(threadIdx.y < TILE_SIZE) {
                if (col + j < N && p*TILE_SIZE + threadIdx.y < K) {
                    ds_B[threadIdx.y][threadIdx.x*TN + j] = weights[(col + j)*K + p*TILE_SIZE + threadIdx.y];
                } else {
                    ds_B[threadIdx.y][threadIdx.x*TN + j] = 0.0f;
                }
            }
        }
        __syncthreads();

        for (int k=0; k<TILE_SIZE; k++) {
            for (int i=0; i<TM; i++) {
                if(row + i < M) {
                    reg_A[i] = ds_A[k][threadIdx.y * TM + i];
                }
            }

            for(int j=0; j<TN; j++) {
                if(col + j < N) {
                    reg_B[j] = ds_B[k][threadIdx.x * TN + j];
                }
            }
            __syncthreads();

            for(int j=0; j<TN; j++) {
                for (int i=0; i<TM; i++) {
                    tie[i][j] += reg_A[i] * reg_B[j];
                }
            }
        }
    }

    for(int i=0; i<TM; i++) {
        for(int j=0; j<TN; j++) {
            if (row + i < M && col + j < N) {
                d_out[(row + i) * N + col + j] = tie[i][j] + bias[row+i];
            }
        }
    }
}

__global__
void kLinear2D_v2(float* input, float* d_out, float* weights, float* bias, int M, int N, int K) {
    // input(N x in_features) @ weights(out_features x in_features).T + bias(out_features)
    float cVal = 0.0f;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float ds_A[BLOCK_SIZE2D][TILE_SIZE];
    __shared__ float ds_B[BLOCK_SIZE2D][TILE_SIZE];

    int phase = (N - 1) / TILE_SIZE + 1;
    for(int p=0; p<phase;p++) {
        if (row < M && threadIdx.x*4 < TILE_SIZE) {
            if(p*TILE_SIZE + threadIdx.x*4 < N) {
                FETCH_FLOAT4(ds_A[threadIdx.y][threadIdx.x*4]) = FETCH_FLOAT4(input[row*N + p*TILE_SIZE + threadIdx.x*4]);
            } else {
                ds_A[threadIdx.y][threadIdx.x*4+0] = 0.0f;
                ds_A[threadIdx.y][threadIdx.x*4+1] = 0.0f;
                ds_A[threadIdx.y][threadIdx.x*4+2] = 0.0f;
                ds_A[threadIdx.y][threadIdx.x*4+3] = 0.0f;
            }
       	}
        
        if (col < K && threadIdx.y*4 < TILE_SIZE) {
            if(p*TILE_SIZE+threadIdx.y*4 < N) {
                FETCH_FLOAT4(ds_B[threadIdx.x][threadIdx.y*4]) = FETCH_FLOAT4(weights[col*N + p*TILE_SIZE + threadIdx.y*4]);
            } else {
                ds_B[threadIdx.x][threadIdx.y*4+0] = 0.0f;
                ds_B[threadIdx.x][threadIdx.y*4+1] = 0.0f;
                ds_B[threadIdx.x][threadIdx.y*4+2] = 0.0f;
                ds_B[threadIdx.x][threadIdx.y*4+3] = 0.0f;
            }
        }

        __syncthreads();
        for (int i=0; i<TILE_SIZE; i++) {
            // constant: ds_A's x , ds_B's x
            cVal += ds_A[threadIdx.y][i] * ds_B[threadIdx.x][i];
        }
        __syncthreads();
    }

    if(row < M && col < K)
        d_out[row*K + col] = cVal + bias[col];
}

__global__
void kLinear2D_v3(float* input, float* d_out, float* weights, float* bias, int M, int N, int K) {
    // input(N x in_features) @ weights(out_features x in_features).T + bias(out_features)
    int row = (blockIdx.y * blockDim.y + threadIdx.y) * TM;
    int col = (blockIdx.x * blockDim.x + threadIdx.x) * TN;

    __shared__ float ds_A[TILE_SIZE][BLOCK_SIZE2D * TM];
    __shared__ float ds_B[TILE_SIZE][BLOCK_SIZE2D * TN];

    float reg_A[TM] = {0.0f};
    float reg_B[TN] = {0.0f};
    float tmp[4] = {0.0f};
    float cVal[TM][TN] = {0.0f};

    int phase = (N - 1) / TILE_SIZE + 1;
    for(int p=0; p<phase;p++) {
        if (row < M && threadIdx.x*4 < TILE_SIZE) {
            if(p*TILE_SIZE + threadIdx.x*4 < N) {
                FETCH_FLOAT4(tmp[0]) = FETCH_FLOAT4(input[row*N + p*TILE_SIZE + threadIdx.x*4]);

                ds_A[threadIdx.x*4 + 0][threadIdx.y] = tmp[0];
                ds_A[threadIdx.x*4 + 1][threadIdx.y] = tmp[1];
                ds_A[threadIdx.x*4 + 2][threadIdx.y] = tmp[2];
                ds_A[threadIdx.x*4 + 3][threadIdx.y] = tmp[3];

            } else {
                ds_A[threadIdx.x*4 + 0][threadIdx.y] = 0.0f;
                ds_A[threadIdx.x*4 + 1][threadIdx.y] = 0.0f;
                ds_A[threadIdx.x*4 + 2][threadIdx.y] = 0.0f;
                ds_A[threadIdx.x*4 + 3][threadIdx.y] = 0.0f;
            }
       	}
        
        if (col < K && threadIdx.y*4 < TILE_SIZE) {
            if(p*TILE_SIZE+threadIdx.y*4 < N) {
                FETCH_FLOAT4(tmp[0]) = FETCH_FLOAT4(weights[col*N + p*TILE_SIZE + threadIdx.y*4]);
                ds_B[threadIdx.y*4+0][threadIdx.x] = tmp[0];
                ds_B[threadIdx.y*4+1][threadIdx.x] = tmp[1];
                ds_B[threadIdx.y*4+2][threadIdx.x] = tmp[2];
                ds_B[threadIdx.y*4+3][threadIdx.x] = tmp[3];
            } else {
                ds_B[threadIdx.x][threadIdx.y*4+0] = 0.0f;
                ds_B[threadIdx.x][threadIdx.y*4+1] = 0.0f;
                ds_B[threadIdx.x][threadIdx.y*4+2] = 0.0f;
                ds_B[threadIdx.x][threadIdx.y*4+3] = 0.0f;
            }

        }

        for(int k=0; k<TILE_SIZE; k++) {
            for(int i=0; i<TM/4; i++) {
                if(row + i*4 < M) {
                    FETCH_FLOAT4(reg_A[i*4]) = FETCH_FLOAT4(ds_A[k][threadIdx.y * TM + i*4]);
                }
                
            }

            for(int j=0; j<TN/4; j++) {
                if(col + j*4 < N) {
                    FETCH_FLOAT4(reg_B[j*4]) = FETCH_FLOAT4(ds_B[k][threadIdx.x * TN + j*4]);
                }
            }
            __syncthreads();

            for(int j=0; j<TN; j++) {
                for (int i=0; i<TM; i++) {
                    cVal[i][j] += reg_A[i] * reg_B[j];
                }
            }
        }
    }

    for(int i=0; i<TM/4; i++) {
        for(int j=0; j<TN; j++) {
            if (row + i*4 < M) {
                FETCH_FLOAT4(tmp[0]) = FETCH_FLOAT4(bias[row + i*4]);
                cVal[i*4+0][j] += tmp[0];
                cVal[i*4+1][j] += tmp[1];
                cVal[i*4+2][j] += tmp[2];
                cVal[i*4+3][j] += tmp[3];
            }
        }
    }

    for(int i=0; i<TM; i++) {
        for(int j=0; j<TN/4; j++) {
            if (row + i < M && col + j*4 < N) {
                FETCH_FLOAT4(d_out[(row + i) * N + col + j*4]) = FETCH_FLOAT4(cVal[i][j*4]);
            }
        }
    }
}


Linear::Linear(std::string prefix, size_t in_features, size_t out_features, bool bias, InitType init_type) {
    this->in_features = in_features;
    this->out_features = out_features;

    DimVector weights_shape = {out_features, in_features};
    this->weights = new Tensor(weights_shape);

    this->weights->initialize(init_type);

    this->prefix = prefix;
    
    if(bias) {
        DimVector bias_shape = {out_features};
        this->bias = new Tensor(bias_shape, this->is_training ? ZERO : NONE);
    }
}

Linear::Linear(size_t in_features, size_t out_features, bool bias, InitType init_type) {
    this->in_features = in_features;
    this->out_features = out_features;

    DimVector weights_shape = {out_features, in_features};
    this->weights = new Tensor(weights_shape);

    this->weights->initialize(init_type);

    if(bias) {
        DimVector bias_shape = {out_features};
        this->bias = new Tensor(bias_shape, this->is_training ? ZERO : NONE);
    }
}

Tensor* Linear::forward(Tensor* data) {
    // data(B x N) @ weightss(M x N).T + bias(M) = output(B x M)
    // reinitializations
    this->reset();
    if(this->is_training)
        this->input = data;
    
    if(data->getDim() != 2) {
        printShape(data->getShape());
        if(this->prefix != "") {
            DEBUG_PRINT("%s\n", this->prefix.c_str());
        }
        ERROR("Not implemented!\n");
    }

    size_t bz = data->getShape()[0];

    this->output = new Tensor({bz, out_features});

    dim3 block(BLOCK_SIZE2D, BLOCK_SIZE2D);
    dim3 grid((out_features - 1)/(BLOCK_SIZE2D)+1, (bz-1)/(BLOCK_SIZE2D)+1);

    // if(bz % 4 == 0 && out_features % 4 == 0 && in_features % 4 == 0) {
    //     kLinear2D_v3<<<grid, block>>>(data->getData(), output->getData(), weights->getData(), bias->getData(), bz, in_features, out_features); CHECK_KERNEL();
    // } else {
    //     kLinear2D<<<grid, block>>>(data->getData(), output->getData(), weights->getData(), bias->getData(), bz, in_features, out_features); CHECK_KERNEL();
    // }

    kLinear2D<<<grid, block>>>(data->getData(), output->getData(), weights->getData(), bias->getData(), bz, in_features, out_features); CHECK_KERNEL();

    if(this->output->getShape()[0] != data->getShape()[0] ||  \
            this->output->getShape()[1] != weights->getShape()[0]) {
                printShape(this->output->getShape());
                ERROR("shape not matched!\n");
            }

    return this->output;
}

Tensor* Linear::backward(Tensor* gradients) {
    return nullptr;
}
