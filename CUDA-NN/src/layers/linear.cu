#include "linear.cuh"

__global__
void kLinear2D(float* input, float* d_out, float* weights, float* bias, int M, int N, int K) {
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
void kLinear2D_v1(float* input, float* d_out, float* weights, float* bias, int M, int N, int K) {
    // float cVal = 0.0f;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row * TM >= M || col * TN >= K) return;

    __shared__ float ds_A[BLOCK_SIZE2D * TM][TILE_SIZE];
    __shared__ float ds_B[BLOCK_SIZE2D * TN][TILE_SIZE];

    float cVal[TM][TN] = {0.0f};

    int phase = (N - 1) / TILE_SIZE + 1;
    for(int p=0; p<phase;p++) { 
        for(int i = 0; i<TM; i++) {
            if (row + i < M && threadIdx.x < TILE_SIZE) {
                if(p*TILE_SIZE + threadIdx.x < N) {
                    ds_A[threadIdx.y * TM + i][threadIdx.x] = input[(row + i)*N + p*TILE_SIZE + threadIdx.x];
                } else {
                    ds_A[threadIdx.y * TM + i][threadIdx.x] = 0.0f;
                }
            }
        }
        
        for(int j=0; j<TN; j++) {
            if ((col+j) < K && threadIdx.y < TILE_SIZE) {
                if(p*TILE_SIZE + threadIdx.y < N) {
                    ds_B[threadIdx.x * TN + j][threadIdx.y] = weights[(col + j)*N + p*TILE_SIZE + threadIdx.y];
                } else {
                    ds_B[threadIdx.x * TN + j][threadIdx.y] = 0.0f;
                }

            }
        }

        __syncthreads();

        for(int i=0; i<TM; i++) {
            for(int j=0; j<TN; j++) {
                for (int k=0; k<TILE_SIZE; k++) {
                    // constant: ds_A's x , ds_B's x
                    cVal[i][j] += ds_A[threadIdx.y*TM+i][k] * ds_B[threadIdx.x*TN+j][k];
                }
            }
        }
        __syncthreads();
    }

    for(int i=0; i<TM; i++) {
        for(int j=0; j<TN; j++) {
            if(row + i < M && col + j < K)
                d_out[(row+i)*K + col+j] = cVal[i][j] + bias[col + j];
        }
    }
}

__global__
void kLinear2D_v2(float* input, float* d_out, float* weights, float* bias, int M, int N, int K) {
    // float cVal = 0.0f;
    int row = (blockIdx.y * blockDim.y + threadIdx.y) * TM;
    int col = (blockIdx.x * blockDim.x + threadIdx.x) * TN;

    int real_row = blockIdx.y * blockDim.y + threadIdx.y;
    int real_col = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float ds_A[BLOCK_SIZE2D * TM][TILE_SIZE];
    __shared__ float ds_B[BLOCK_SIZE2D * TN][TILE_SIZE];
    // __shared__ float ds_bias[BLOCK_SIZE2D * TN];

    float cVal[TM][TN] = {0.0f};

    int phase = (N - 1) / TILE_SIZE + 1;
    for(int p=0; p<phase;p++) { 
        if (real_row < M && threadIdx.x < TILE_SIZE) {
            if(p*TILE_SIZE + threadIdx.x < N) {
                ds_A[threadIdx.y][threadIdx.x] = input[real_row * N + p*TILE_SIZE + threadIdx.x];
            } else {
                ds_A[threadIdx.y][threadIdx.x] = 0.0f;
            }
        }
    
        if (real_col < K && threadIdx.y < TILE_SIZE) {
            if(p*TILE_SIZE + threadIdx.y < N) {
                ds_B[threadIdx.x][threadIdx.y] = weights[real_col * N + p*TILE_SIZE + threadIdx.y];
            } else {
                ds_B[threadIdx.x][threadIdx.y] = 0.0f;
            }

        }

        __syncthreads();

        for(int i=0; i<TM; i++) {
            for(int j=0; j<TN; j++) {
                for (int k=0; k<TILE_SIZE; k++) {
                    // constant: ds_A's x , ds_B's x
                    if(row + i < M && col + j < K)
                        cVal[i][j] += ds_A[threadIdx.y*TM+i][k] * ds_B[threadIdx.x*TN+j][k];
                }
            }
        }
        __syncthreads();
    }

    for(int i=0; i<TM; i++) {
        for(int j=0; j<TN; j++) {
            if(row + i < M && col + j < K)
                d_out[(row+i)*K + col+j] = cVal[i][j] + bias[col + j];
        }
    }
}

__global__
void kLinear3D(float* input, float* d_out, float* weights, float* bias, int M, int N, int K) {
    float cVal = 0.0f;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float ds_A[BLOCK_SIZE2D][TILE_SIZE];
    __shared__ float ds_B[BLOCK_SIZE2D][TILE_SIZE];
    __shared__ float ds_bias[BLOCK_SIZE2D];

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
        if(threadIdx.y == 0)
            ds_bias[threadIdx.x] = bias[col];

        __syncthreads();
        for (int i=0; i<TILE_SIZE; i++) {
            // constant: ds_A's x , ds_B's x
            cVal += ds_A[threadIdx.y][i] * ds_B[threadIdx.x][i];
        }
        __syncthreads();
    }

    if(row < M && col < K)
        d_out[row*K + col] = cVal + ds_bias[threadIdx.x];
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
        this->bias = new Tensor(bias_shape, ZERO);
        this->bias->initialize(0.0f);
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
        this->bias = new Tensor(bias_shape, ZERO);
        this->bias->initialize(0.0f);
    }
}

Linear::~Linear() {
    delete weights;
    delete bias;
    delete input;
    delete output;
    delete outputBackward;
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
    dim3 grid((out_features - 1)/(BLOCK_SIZE2D)+1, (bz-1)/(BLOCK_SIZE2D) + 1);

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
