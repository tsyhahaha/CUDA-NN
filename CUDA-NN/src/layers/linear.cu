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

__global__ void kBackprop_to_weights_and_bias(
    float *gradients, float* in, float *d_weights, float *d_bias, int N, int C_in, int C_out) {
    // gradients(N, C_out).T @ input(N, C_in) = d_weights(C_out, C_in)
    // gradients(N, C_out) -> d_bias(C_out)
    float cVal = 0.0f;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row >= C_out || col >= C_in) return;

    __shared__ float ds_A[BLOCK_SIZE2D][TILE_SIZE];
    __shared__ float ds_B[TILE_SIZE][BLOCK_SIZE2D];

    int phase = (N - 1) / TILE_SIZE + 1;
    for(int p=0; p<phase;p++) {
        if(threadIdx.x < TILE_SIZE) { 
            if (row < C_out && p*TILE_SIZE + threadIdx.x < N) {
                // if(threadIdx.y == 12 && blockIdx.x == 14) {
                //     printf("N=%d, C_out=%d, gradients[%d][%d]\n", N, C_out, p*TILE_SIZE + threadIdx.x, row);
                // }
                ds_A[threadIdx.y][threadIdx.x] = gradients[(p*TILE_SIZE + threadIdx.x)*C_out + row];
            } else if(threadIdx.y < BLOCK_SIZE2D && threadIdx.x < TILE_SIZE) {
                ds_A[threadIdx.y][threadIdx.x] = 0.0f;
            }
        }

        if(threadIdx.y < TILE_SIZE) {
            if(col < C_in && p*TILE_SIZE + threadIdx.y < N) {
                if(threadIdx.y == 12 && blockIdx.x == 14) {
                    printf("in[%d][%d]\n", p*TILE_SIZE + threadIdx.y, col);
                }
                ds_B[threadIdx.y][threadIdx.x] = in[(p*TILE_SIZE + threadIdx.y)*C_in + col];
            } else if(threadIdx.y < TILE_SIZE && threadIdx.x < BLOCK_SIZE2D) {
                ds_B[threadIdx.y][threadIdx.x] = 0.0f;
            }
        }

        __syncthreads();
        for (int i=0; i<TILE_SIZE; i++) {
            // constant: ds_A's x , ds_B's y
            cVal += ds_A[threadIdx.y][i] * ds_B[i][threadIdx.x];
        }
        __syncthreads();

        // accumulate to bias
        for(int stride=TILE_SIZE/2; stride>0; stride>>=1) {
            if(threadIdx.x < stride && threadIdx.x + stride + p*TILE_SIZE < N) {
                ds_A[threadIdx.y][threadIdx.x] = ds_A[threadIdx.y][threadIdx.x] + ds_A[threadIdx.y][threadIdx.x + stride];
                __syncthreads();
            }
        }
        if(threadIdx.x == 0) {
            atomicAdd(&d_bias[row], ds_A[threadIdx.y][0]);
        }
    }

    if (row < C_out && col < C_in)
        atomicAdd(&d_weights[row*C_in + col], cVal);
}

Linear::Linear(std::string prefix, size_t in_features, size_t out_features, bool bias) {
    this->in_features = in_features;
    this->out_features = out_features;

    DimVector weights_shape = {out_features, in_features};
    this->weights = new Tensor(weights_shape, NONE);

    this->prefix = prefix;
    
    if(bias) {
        DimVector bias_shape = {out_features};
        this->bias = new Tensor(bias_shape, NONE);
    }
}

Linear::Linear(size_t in_features, size_t out_features, bool bias) {
    this->in_features = in_features;
    this->out_features = out_features;

    DimVector weights_shape = {out_features, in_features};
    this->weights = new Tensor(weights_shape, NONE);

    if(bias) {
        DimVector bias_shape = {out_features};
        this->bias = new Tensor(bias_shape, NONE);
    }
}

void Linear::init_weights() {
    float sqrt_k = 1.0f/(sqrt(in_features));
    this->weights->initialize(KAIMING, sqrt_k);
    DEBUG_PRINT("Linear init weights: KAIMING\n");
    if(bias) {
        this->bias->initialize(KAIMING, sqrt_k);
        DEBUG_PRINT("Linear init bias: KAIMING\n");
    }
}

Tensor* Linear::forward(Tensor* data) {
    DEBUG_PRINT("[Linear] %sforward\n", this->prefix.c_str());

    // data(B x N) @ weights(M x N).T + bias(M) = output(B x M)
    // reinitializations
    size_t bz = data->getShape()[0];
    DimVector shape_o = {bz, out_features};

    if(output==nullptr) {
        this->output = new Tensor({bz, out_features});
    }
    this->output->reset(shape_o);

    if(this->is_training)
        this->input = data;
    
    if(data->getDim() != 2) {
        printShape(data->getShape());
        if(this->prefix != "") {
            DEBUG_PRINT("%s\n", this->prefix.c_str());
        }
        ERROR("Not implemented!\n");
    }

    dim3 block(BLOCK_SIZE2D, BLOCK_SIZE2D);
    dim3 grid((out_features - 1)/(BLOCK_SIZE2D)+1, (bz-1)/(BLOCK_SIZE2D)+1);

    kLinear2D<<<grid, block>>>(data->getData(), output->getData(), weights->getData(), bias->getData(), bz, in_features, out_features); CHECK_KERNEL();

    if(this->output->getShape()[0] != data->getShape()[0] ||  \
            this->output->getShape()[1] != weights->getShape()[0]) {
                printShape(data->getShape());
                printShape(this->output->getShape());
                ERROR("shape not matched!\n");
            }


    return this->output;
}

Linear* Linear::train() {
    BaseLayer::train();
    size_t bz = Configurer::batch_size;
    if(!d_in) {
        this->d_in = new Tensor({bz, in_features});
    } this->d_in->reset({bz, in_features});
    return this;
}

Tensor* Linear::backward(Tensor* gradients) {
    DEBUG_PRINT("[Linear] %sbackward\n", this->prefix.c_str());

    int N = gradients->getSize(0), C_in = in_features, C_out = out_features;
    // gradients->transpose(); // (C_out x B)
    // // d_out(C_out x B) @ input(C_in x B).T = (C_out x C_in)
    // gradients->matmul(this->d_weights, this->input);
    // gradients->transpose(); // (B x C_out)
    // gradients->sumToDim(d_bias, 1); // (B x C_out)->(C_out)
    // accumulate grads
    // this->weights->acc_grads(d_weights);
    // this->bias->acc_grads(d_bias);


    dim3 block(BLOCK_SIZE2D, BLOCK_SIZE2D);
    dim3 grid((in_features - 1)/(BLOCK_SIZE2D)+1, (out_features-1)/(BLOCK_SIZE2D)+1);


    Tensor* d_weights = weights->getGradsAcc();
    Tensor* d_bias = bias->getGradsAcc();

    kBackprop_to_weights_and_bias<<<grid, block>>>(gradients->getData(), input->getData(), d_weights->getData(), d_bias->getData(), N, C_in, C_out); CHECK_KERNEL();

    // d_out(B x C_out) @ weights(C_out x C_in) = d_in(B x C_in)
    gradients->matmul(this->d_in, this->weights);

    return d_in;
}
