#include "conv1d.cuh"

#define PRINT_IDX() printf("block(%d,%d,%d) thread(%d,%d,%d) ", blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z)

__global__
void kConv1d(float* d_in, float* d_out, float* weights, float* bias, int C_in, int C_out, int L, int N) {
    // weights(C_out x C_in) @ d_in(B x C_in x L) + bias(C_out)= (B x C_out x L)
    float cVal = 0.0f;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float ds_A[BLOCK_SIZE3D][TILE_SIZE];
    __shared__ float ds_B[TILE_SIZE][BLOCK_SIZE3D];

    for(int b=0; b<N; b++) {
        // batch b
	    cVal = 0.0f;
        int phase = (C_in - 1) / TILE_SIZE + 1;
        for(int p=0; p<phase;p++) {
            if (row < C_out && threadIdx.x < TILE_SIZE) {
                if(p*TILE_SIZE + threadIdx.x < C_in) {
                    ds_A[threadIdx.y][threadIdx.x] = weights[row*C_in + p*TILE_SIZE + threadIdx.x];
                } else {
                    ds_A[threadIdx.y][threadIdx.x] = 0.0f;
                }
            }

            if(col < L && threadIdx.y < TILE_SIZE) {
                if(p*TILE_SIZE + threadIdx.y < C_in){
                    ds_B[threadIdx.y][threadIdx.x] = d_in[b*C_in*L + (p*TILE_SIZE + threadIdx.y)*L + col];
                } else {
                    ds_B[threadIdx.y][threadIdx.x] = 0.0f;
                }
            }

            __syncthreads();
            for (int i=0; i<TILE_SIZE; i++) {
                // constant: ds_A's x , ds_B's y
                cVal += ds_A[threadIdx.y][i] * ds_B[i][threadIdx.x];
            }
            __syncthreads();
        }
        if(row < C_out && col < L)
            d_out[b*C_out*L + row*L + col] = cVal + bias[row];
    }
}

__global__
void kConv1d_v1(float* d_in, float* d_out, float* weights, float* bias, int C_in, int C_out, int L, int N) {
    // weights(C_out x C_in) @ d_in(B x C_in x L) + bias(C_out)= (B x C_out x L)
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int batch = blockIdx.z * blockDim.z + threadIdx.z;

    __shared__ float ds_A[BLOCK_SIZE3D][TILE_SIZE];
    __shared__ float ds_B[BATCH_BASE][TILE_SIZE][BLOCK_SIZE3D];

    if(batch >= N) return;

    int offset_in  = batch * C_in  * L;
    int offset_out = batch * C_out * L;

    float cVal = 0.0f;
    int phase = (C_in - 1) / TILE_SIZE + 1;
    for(int p=0; p<phase;p++) {
        if (threadIdx.z == 0 && row < C_out && threadIdx.x < TILE_SIZE) {
            if(p*TILE_SIZE + threadIdx.x < C_in) {
                ds_A[threadIdx.y][threadIdx.x] = weights[row*C_in + p*TILE_SIZE + threadIdx.x];
            } else {
                ds_A[threadIdx.y][threadIdx.x] = 0.0f;
            }
        }

        if(col < L && threadIdx.y < TILE_SIZE) {
            if(p*TILE_SIZE + threadIdx.y < C_in){
                ds_B[threadIdx.z][threadIdx.y][threadIdx.x] = d_in[offset_in + (p*TILE_SIZE + threadIdx.y)*L + col];
            } else {
                ds_B[threadIdx.z][threadIdx.y][threadIdx.x] = 0.0f;
            }
        }
        __syncthreads();

        for (int i=0; i<TILE_SIZE; i++) {
            // constant: ds_A's x , ds_B's y
            cVal += ds_A[threadIdx.y][i] * ds_B[threadIdx.z][i][threadIdx.x];
        }
        __syncthreads();
    }
    if(row < C_out && col < L)
        d_out[offset_out + row*L + col] = cVal + bias[row];
}

__global__
void kConv1d_v2(float* d_in, float* d_out, float* weights, float* bias, int C_in, int C_out, int L, int N) {
    // weights(C_out x C_in) @ d_in(B x C_in x L) + bias(C_out)= (B x C_out x L)
    int row = (blockIdx.y * blockDim.y + threadIdx.y)*TM;
    int col = (blockIdx.x * blockDim.x + threadIdx.x)*TN;
    int batch = blockIdx.z * blockDim.z + threadIdx.z;

    if(batch >= N) return;

    __shared__ float ds_A[BLOCK_SIZE3D * TM][TILE_SIZE];
    __shared__ float ds_B[BATCH_BASE][TILE_SIZE][BLOCK_SIZE3D * TN];

    float tie[TM][TN] = {0.0f};
    float reg_A[TM] = {0.0f};
    float reg_B[TN] = {0.0f};

    uint offset_in = batch * C_in * L;
    uint offset_out = batch * C_out * L;

    int phase = (C_in - 1) / TILE_SIZE + 1;
    for(int p=0; p<phase;p++) {
        for(int i=0; i<TM; i++) {
            if(threadIdx.z == 0 && threadIdx.x < TILE_SIZE) {
                if(row+i < C_out && p*TILE_SIZE + threadIdx.x < C_in) {
                    ds_A[threadIdx.y*TM+i][threadIdx.x] = weights[(row+i) * C_in + p*TILE_SIZE + threadIdx.x];
                } else {
                    ds_A[threadIdx.y*TM+i][threadIdx.x] = 0.0f;
                }
            }
        }

        for(int j=0; j<TN; j++) {
            if(threadIdx.y < TILE_SIZE) {
                if (col + j < L && p*TILE_SIZE + threadIdx.y < C_in) {
                    ds_B[threadIdx.z][threadIdx.y][threadIdx.x*TN + j] = d_in[offset_in + (p*TILE_SIZE + threadIdx.y)*L + col + j];
                } else {
                    ds_B[threadIdx.z][threadIdx.y][threadIdx.x*TN + j] = 0.0f;
                }
            }
        }
        __syncthreads();
        

        for (int k=0; k<TILE_SIZE; k++) {
            for (int i=0; i<TM; i++) {
                if(row + i < C_out) {
                    reg_A[i] = ds_A[threadIdx.y * TM + i][k];
                } else {
                    reg_A[i] = 0.0f;
                }
            }

            for(int j=0; j<TN; j++) {
                if(col + j < L) {
                    reg_B[j] = ds_B[threadIdx.z][k][threadIdx.x * TN + j];
                } else {
                    reg_B[j] = 0.0f;
                }
            }
            __syncthreads();

            for(int j=0; j<TN; j++) {
                for (int i=0; i<TM; i++) {
                    // if(col + j < L && row + i < C_out)
                    tie[i][j] += reg_A[i] * reg_B[j];
                }
            }
            __syncthreads();
        }
    }

    for(int i=0; i<TM; i++) {
        for(int j=0; j<TN; j++) {
            if (row + i < C_out && col + j < L) {
                d_out[offset_out + (row + i) * L + col + j] = tie[i][j] + bias[row + i];
            }
        }
    }
}

__global__
void kConv1d_v3(float* d_in, float* d_out, float* weights, float* bias, int C_in, int C_out, int L, int N) {
    // weights(C_out x C_in) @ d_in(B x C_in x L) + bias(C_out)= (B x C_out x L)
    int row = (blockIdx.y * blockDim.y + threadIdx.y)*TM;
    int col = (blockIdx.x * blockDim.x + threadIdx.x)*TN;
    int batch = blockIdx.z * blockDim.z + threadIdx.z;

    if(batch >= N) return;

    __shared__ float ds_w[TILE_SIZE][BLOCK_SIZE3D * TM];
    __shared__ float ds_in[BATCH_BASE][TILE_SIZE][BLOCK_SIZE3D * TN];

    float tie[TM][TN] = {0.0f};
    float reg_w[TM] = {0.0f};
    float reg_in[TN] = {0.0f};
    float tmp[4] = {0.0f};

    uint offset_in = batch * C_in * L;
    uint offset_out = batch * C_out * L;

    int phase = (C_in - 1) / TILE_SIZE + 1;
    for(int p=0; p<phase;p++) {
        for(int i=0; i<TM; i++) {
            if(threadIdx.z == 0 && threadIdx.x*4 < TILE_SIZE) {
                if(row+i < C_out && p*TILE_SIZE + threadIdx.x*4 < C_in) {
                    FETCH_FLOAT4(tmp[0]) = FETCH_FLOAT4(weights[(row+i) * C_in + p*TILE_SIZE + threadIdx.x*4]);

                    ds_w[threadIdx.x*4+0][threadIdx.y*TM+i] = tmp[0];
                    ds_w[threadIdx.x*4+1][threadIdx.y*TM+i] = tmp[1];
                    ds_w[threadIdx.x*4+2][threadIdx.y*TM+i] = tmp[2];
                    ds_w[threadIdx.x*4+3][threadIdx.y*TM+i] = tmp[3];

                    // ds_w[threadIdx.y*TM+i][threadIdx.x] = weights[(row+i) * C_in + p*TILE_SIZE + threadIdx.x];
                } else {
                    ds_w[threadIdx.x*4+0][threadIdx.y*TM+i] = 0.0f;
                    ds_w[threadIdx.x*4+1][threadIdx.y*TM+i] = 0.0f;
                    ds_w[threadIdx.x*4+2][threadIdx.y*TM+i] = 0.0f;
                    ds_w[threadIdx.x*4+3][threadIdx.y*TM+i] = 0.0f;

                    // ds_w[threadIdx.y*TM+i][threadIdx.x] = 0.0f;
                }
            }
        }
        

        for(int j=0; j<TN/4; j++) {
            if(threadIdx.y < TILE_SIZE) {
                if (col + j*4 < L && p*TILE_SIZE + threadIdx.y < C_in) {
                    FETCH_FLOAT4(ds_in[threadIdx.z][threadIdx.y][threadIdx.x*TN + j*4]) = FETCH_FLOAT4(d_in[offset_in + (p*TILE_SIZE + threadIdx.y)*L + col + j*4]);
                } else {
                    ds_in[threadIdx.z][threadIdx.y][threadIdx.x*TN + j*4 + 0] = 0.0f;
                    ds_in[threadIdx.z][threadIdx.y][threadIdx.x*TN + j*4 + 1] = 0.0f;
                    ds_in[threadIdx.z][threadIdx.y][threadIdx.x*TN + j*4 + 2] = 0.0f;
                    ds_in[threadIdx.z][threadIdx.y][threadIdx.x*TN + j*4 + 3] = 0.0f;
                }
            }
        }
        __syncthreads();

        for (int k=0; k<TILE_SIZE; k++) {
            for (int i=0; i<TM/4; i++) {
                if(row + i*4 < C_out) {
                    FETCH_FLOAT4(reg_w[i*4]) = FETCH_FLOAT4(ds_w[k][threadIdx.y * TM + i*4]);
                }
            }

            for(int j=0; j<TN/4; j++) {
                if(col + j*4 < L) {
                    FETCH_FLOAT4(reg_in[j*4]) = FETCH_FLOAT4(ds_in[threadIdx.z][k][threadIdx.x * TN + j*4]);
                }
            }
            __syncthreads();

            for(int j=0; j<TN; j++) {
                for (int i=0; i<TM; i++) {
                    tie[i][j] += reg_w[i] * reg_in[j];
                }
            }
        }
    }

    for(int i=0; i<TM/4; i++) {
        for(int j=0; j<TN; j++) {
            if (row + i*4 < C_out) {
                FETCH_FLOAT4(tmp[0]) = FETCH_FLOAT4(bias[row + i*4]);
                tie[i*4+0][j] += tmp[0];
                tie[i*4+1][j] += tmp[1];
                tie[i*4+2][j] += tmp[2];
                tie[i*4+3][j] += tmp[3];
            }
        }
    }

    for(int i=0; i<TM; i++) {
        for(int j=0; j<TN/4; j++) {
            if (row + i < C_out && col + j*4 < L) {
                FETCH_FLOAT4(d_out[offset_out + (row + i) * L + col + j*4]) = FETCH_FLOAT4(tie[i][j*4]);
            }
        }
    }
}

__global__
void kConv1d_back_v2(float* gradients, float* d_in, float* weights, int C_in, int C_out, int L, int N) {
    // W.T(C_in, C_out) @ gradients(N, C_out, L) @  = d_in(N, C_in, L)
    // weights(C_out x C_in) @ input(B x C_in x L) + bias(C_out)= (B x C_out x L)
    int row = (blockIdx.y * blockDim.y + threadIdx.y)*TM;
    int col = (blockIdx.x * blockDim.x + threadIdx.x)*TN;
    int batch = blockIdx.z * blockDim.z + threadIdx.z;

    if(batch >= N) return;

    __shared__ float ds_A[BLOCK_SIZE3D * TM][TILE_SIZE];
    __shared__ float ds_B[BATCH_BASE][TILE_SIZE][BLOCK_SIZE3D * TN];

    float tie[TM][TN] = {0.0f};
    float reg_A[TM] = {0.0f};
    float reg_B[TN] = {0.0f};

    uint offset_in = batch * C_out * L;
    uint offset_out = batch * C_in * L;

    int phase = (C_out - 1) / TILE_SIZE + 1;
    for(int p=0; p<phase;p++) {
        for(int i=0; i<TM; i++) {
            if(threadIdx.z == 0 && threadIdx.x < TILE_SIZE) {
                if(row+i < C_in && p*TILE_SIZE + threadIdx.x < C_out) {
                    ds_A[threadIdx.y*TM+i][threadIdx.x] = weights[(row+i) * C_out + p*TILE_SIZE + threadIdx.x];
                } else {
                    ds_A[threadIdx.y*TM+i][threadIdx.x] = 0.0f;
                }
            }
        }

        for(int j=0; j<TN; j++) {
            if(threadIdx.y < TILE_SIZE) {
                if (col + j < L && p*TILE_SIZE + threadIdx.y < C_in) {
                    ds_B[threadIdx.z][threadIdx.y][threadIdx.x*TN + j] = d_in[offset_in + (p*TILE_SIZE + threadIdx.y)*L + col + j];
                } else {
                    ds_B[threadIdx.z][threadIdx.y][threadIdx.x*TN + j] = 0.0f;
                }
            }
        }
        __syncthreads();
        

        for (int k=0; k<TILE_SIZE; k++) {
            for (int i=0; i<TM; i++) {
                if(row + i < C_in) {
                    reg_A[i] = ds_A[threadIdx.y * TM + i][k];
                } else {
                    reg_A[i] = 0.0f;
                }
            }

            for(int j=0; j<TN; j++) {
                if(col + j < L) {
                    reg_B[j] = ds_B[threadIdx.z][k][threadIdx.x * TN + j];
                } else {
                    reg_B[j] = 0.0f;
                }
            }
            __syncthreads();

            for(int j=0; j<TN; j++) {
                for (int i=0; i<TM; i++) {
                    // if(col + j < L && row + i < C_out)
                    tie[i][j] += reg_A[i] * reg_B[j];
                }
            }
            __syncthreads();
        }
    }

    for(int i=0; i<TM; i++) {
        for(int j=0; j<TN; j++) {
            if (row + i < C_in && col + j < L) {
                d_in[offset_out + (row + i) * L + col + j] = tie[i][j];
            }
        }
    }
}


__global__
void kConv1d_back_v3(float* gradients, float* d_in, float* weights, int C_in, int C_out, int L, int N) {
    int row = (blockIdx.y * blockDim.y + threadIdx.y)*TM;
    int col = (blockIdx.x * blockDim.x + threadIdx.x)*TN;
    int batch = blockIdx.z * blockDim.z + threadIdx.z;

    if(batch >= N) return;

    __shared__ float ds_w[TILE_SIZE][BLOCK_SIZE3D * TM];
    __shared__ float ds_in[BATCH_BASE][TILE_SIZE][BLOCK_SIZE3D * TN];

    float tie[TM][TN] = {0.0f};
    float reg_w[TM] = {0.0f};
    float reg_in[TN] = {0.0f};

    uint offset_in = batch * C_out * L;
    uint offset_out = batch * C_in * L;

    int phase = (C_out - 1) / TILE_SIZE + 1;
    for(int p=0; p<phase;p++) {
        for(int i=0; i<TM/4; i++) {
            if(threadIdx.z == 0 && threadIdx.x < TILE_SIZE) {
                if(row + i*4 < C_out && p*TILE_SIZE + threadIdx.x < C_in) {
                    FETCH_FLOAT4(ds_w[threadIdx.x][threadIdx.y*TM+i*4]) = FETCH_FLOAT4(weights[(row+i) * C_out + p*TILE_SIZE + threadIdx.x*4]);
                } else {
                    ds_w[threadIdx.x][threadIdx.y*TM+i*4+0] = 0.0f;
                    ds_w[threadIdx.x][threadIdx.y*TM+i*4+1] = 0.0f;
                    ds_w[threadIdx.x][threadIdx.y*TM+i*4+2] = 0.0f;
                    ds_w[threadIdx.x][threadIdx.y*TM+i*4+3] = 0.0f;
                }
            }
        }
        

        for(int j=0; j<TN/4; j++) {
            if(threadIdx.y < TILE_SIZE) {
                if (col + j*4 < L && p*TILE_SIZE + threadIdx.y < C_in) {
                    FETCH_FLOAT4(ds_in[threadIdx.z][threadIdx.y][threadIdx.x*TN + j*4]) = FETCH_FLOAT4(gradients[offset_in + (p*TILE_SIZE + threadIdx.y)*L + col + j*4]);
                } else {
                    ds_in[threadIdx.z][threadIdx.y][threadIdx.x*TN + j*4 + 0] = 0.0f;
                    ds_in[threadIdx.z][threadIdx.y][threadIdx.x*TN + j*4 + 1] = 0.0f;
                    ds_in[threadIdx.z][threadIdx.y][threadIdx.x*TN + j*4 + 2] = 0.0f;
                    ds_in[threadIdx.z][threadIdx.y][threadIdx.x*TN + j*4 + 3] = 0.0f;
                }
            }
        }
        __syncthreads();

        for (int k=0; k<TILE_SIZE; k++) {
            for (int i=0; i<TM/4; i++) {
                if(row + i*4 < C_in) {
                    FETCH_FLOAT4(reg_w[i*4]) = FETCH_FLOAT4(ds_w[k][threadIdx.y * TM + i*4]);
                }
            }

            for(int j=0; j<TN/4; j++) {
                if(col + j*4 < L) {
                    FETCH_FLOAT4(reg_in[j*4]) = FETCH_FLOAT4(ds_in[threadIdx.z][k][threadIdx.x * TN + j*4]);
                }
            }
            __syncthreads();

            for(int j=0; j<TN; j++) {
                for (int i=0; i<TM; i++) {
                    tie[i][j] += reg_w[i] * reg_in[j];
                }
            }
        }
    }

    for(int i=0; i<TM; i++) {
        for(int j=0; j<TN/4; j++) {
            if (row + i < C_in && col + j*4 < L) {
                FETCH_FLOAT4(d_in[offset_out + (row + i) * L + col + j*4]) = FETCH_FLOAT4(tie[i][j*4]);
            }
        }
    }
}

__global__
void kBackprop_to_weights_and_bias(
    float* gradients, float* input, float* d_weights, float* d_bias, size_t N, size_t C_out, size_t C_in, size_t L
) {
    // A (N x C_out x L) @ B (N x L x C_in) = C (B x C_out x C_in)
    float cVal = 0.0f;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int batch = blockIdx.z * blockDim.z + threadIdx.z;

    // if(batch >= N || row >= C_out || col >= C_in) return;
    if(batch >= N) return;

    __shared__ float ds_A[BATCH_BASE][BLOCK_SIZE2D][TILE_SIZE];
    __shared__ float ds_B[BATCH_BASE][TILE_SIZE][BLOCK_SIZE2D];

    int offset1 = batch * C_out * L;
    int offset2 = batch * L * C_in;
    int phase = (L - 1) / TILE_SIZE + 1;
    for(int p=0; p<phase; p++) {
        if (threadIdx.x < TILE_SIZE) {
            if(row < C_out && p*TILE_SIZE + threadIdx.x < L) {
                ds_A[threadIdx.z][threadIdx.y][threadIdx.x] = gradients[offset1 + row*L + p*TILE_SIZE + threadIdx.x];
            } else {
                ds_A[threadIdx.z][threadIdx.y][threadIdx.x] = 0.0f;
            }
        }

        if(threadIdx.y < TILE_SIZE) {
            if(col < C_in && p*TILE_SIZE + threadIdx.y < L) {
                ds_B[threadIdx.z][threadIdx.y][threadIdx.x] = input[offset2 + (p*TILE_SIZE + threadIdx.y)*C_in + col];
            } else {
                ds_B[threadIdx.z][threadIdx.y][threadIdx.x] = 0.0f;
            }
        }

        __syncthreads();
        for (int i=0; i<TILE_SIZE; i++) {
            // constant: ds_A's x , ds_B's y
            cVal += ds_A[threadIdx.z][threadIdx.y][i] * ds_B[threadIdx.z][i][threadIdx.x];
        }
        __syncthreads();

        // accumulate to bias
        for(int stride=TILE_SIZE/2; stride>0; stride>>=1) {
            if(threadIdx.x < stride && threadIdx.x + stride < TILE_SIZE && threadIdx.x + stride + p*TILE_SIZE < L) {
                ds_A[threadIdx.z][threadIdx.y][threadIdx.x] = ds_A[threadIdx.z][threadIdx.y][threadIdx.x] + ds_A[threadIdx.z][threadIdx.y][threadIdx.x + stride];
            }
        }

        if(threadIdx.x == 0) {
            // printf("[atomicAdd] to bias\n");
            atomicAdd(&d_bias[row], ds_A[threadIdx.z][threadIdx.y][0]);
        }
    }

    if (row < C_out && col < C_in) {
        // printf("[atomicAdd] to weights\n");
        atomicAdd(&d_weights[row*C_in + col], cVal);
    }
}

Conv1d::Conv1d(std::string prefix, size_t in_channels, size_t out_channels, size_t kernel_size, bool bias) {
    this->in_channels = in_channels;
    this->out_channels = out_channels;
    this->kernel_size = this->kernel_size;

    this->prefix = prefix;

    if(kernel_size != 1) {
        perror("Not implemented!");
    }

    if(bias) {
        this->bias = new Tensor({out_channels}, NONE);
    }

    this->weights = new Tensor({out_channels, in_channels}, NONE);
}


Conv1d::Conv1d(size_t in_channels, size_t out_channels, size_t kernel_size, bool bias) {
    this->in_channels = in_channels;
    this->out_channels = out_channels;
    this->kernel_size = this->kernel_size;

    if(kernel_size != 1) {
        perror("Not implemented!");
    }

    if(bias) {
        this->bias = new Tensor({out_channels}, NONE);
    } else 
        this->bias = nullptr;

    this->weights = new Tensor({out_channels, in_channels}, NONE);
}

void Conv1d::init_weights() {
    float sqrt_k = 1.0f/(sqrt(in_channels));
    this->weights->initialize(is_training ? KAIMING:NONE, sqrt_k);

    DEBUG_PRINT("Conv1d init weights: KAIMING\n");
    if(bias) {
        this->bias->initialize(is_training ? KAIMING:NONE, sqrt_k);
        DEBUG_PRINT("Conv1d init bias:  KAIMING\n");
    }
}

Conv1d* Conv1d::train() {
    // BaseLayer::train();
    this->is_training = true;
    if(weights)
        this->weights->train();
    if(bias)
        this->bias->train();

    size_t bz = Configurer::batch_size;
    size_t l = Configurer::cropping_size;
    
    if(!d_in) {
        d_in = new Tensor({bz, in_channels, l});
    }
    return this;
}

/*
The implementation of Conv1d simplifies for Pointnet.

im2col to accelerate conv1d op
 - input: (N x C_in x L_in) or (C_in, L_in)
 - weights: (C_out x C_in)
 - bias: (C_out)
 - output(reshape): (N x C_out x L_in) or (C_out x L_in)
*/
Tensor* Conv1d::forward(Tensor* data) {
    DEBUG_PRINT("[Conv1d] %sforward\n", this->prefix.c_str());

    size_t dim = data->getDim();
    DimVector shape_in = data->getShape();
    DimVector shape_o;

    if(dim == 2) {
        shape_o = {out_channels, shape_in[1]};
    } else if(dim==3) {
        shape_o = {shape_in[0], out_channels, shape_in[2]};
    } else {
        ERROR("Not implemented!\n");
    }
    
    if(output ==nullptr) {
        this->output = new Tensor(shape_o);
    }
    this->output->reset(shape_o);

    if(this->is_training)
        this->input = data;

    
    if(dim == 2) {
        size_t C_in = in_channels, C_out = out_channels, L = shape_in[1];
        dim3 block(BLOCK_SIZE2D, BLOCK_SIZE2D);
        dim3 grid((L-1)/BLOCK_SIZE2D+1, (C_out-1)/BLOCK_SIZE2D+1);

        kConv1d<<<grid, block>>>(data->getData(), output->getData(), weights->getData(), bias->getData(), C_in, C_out, L, 1); CHECK_KERNEL();
    } else if(dim == 3) {
        size_t C_in = in_channels, C_out = out_channels;
        size_t L = shape_in[2], N = shape_in[0];

        dim3 block(BLOCK_SIZE3D, BLOCK_SIZE3D, BATCH_BASE);
        dim3 grid((L-1)/(BLOCK_SIZE3D*TN)+1, (C_out-1)/(BLOCK_SIZE3D*TM)+1, (N-1)/BATCH_BASE + 1);

        // kConv1d_v2<<<grid, block>>>(data->getData(), output->getData(), weights->getData(), bias->getData(), C_in, C_out, L, N); 

        if(C_in % 4 == 0 && L % 4 == 0) {
            kConv1d_v3<<<grid, block>>>(data->getData(), output->getData(), weights->getData(), bias->getData(), C_in, C_out, L, N); CHECK_KERNEL();
        } else {
            // DEBUG_PRINT("C_in=%d, C_out=%d, N=%d\n", C_in, C_out, N);
            kConv1d_v2<<<grid, block>>>(data->getData(), output->getData(), weights->getData(), bias->getData(), C_in, C_out, L, N); CHECK_KERNEL();
        }
    } else {
        ERROR("Dimension not allowed!\n");
    }

    return this->output;
}

Tensor* Conv1d::backward(Tensor* gradients) {
    DEBUG_PRINT("[Conv1d] %sbackward\n", this->prefix.c_str());
    DimVector shape_in = input->getShape();
    size_t N = input->getSize(0), L = Configurer::cropping_size;
    if(input->getSize(-1) == L) {
        input->transpose(-2, -1);
    }
    // gradients->bmm(d_weights, input);   // (N,C_out,L)x(N,L,C_in)

    dim3 block(BLOCK_SIZE2D, BLOCK_SIZE2D, BATCH_BASE);
    dim3 grid((in_channels-1)/BLOCK_SIZE2D+1, (out_channels-1)/BLOCK_SIZE2D+1, (N-1)/BATCH_BASE+1);

    Tensor* d_weights_acc = weights->getGradsAcc();
    Tensor* d_bias_acc = bias->getGradsAcc();

    // printShape(gradients->getShape());
    // printShape(input->getShape());
    // printShape(d_weights_acc->getShape());
    // printShape(d_bias_acc->getShape());
    // DEBUG_PRINT("(%d, %d, %d, %d)\n", N, out_channels, in_channels, L);

    kBackprop_to_weights_and_bias<<<grid, block>>>(gradients->getData(), input->getData(), d_weights_acc->getData(), d_bias_acc->getData(), N, out_channels, in_channels, L); CHECK_KERNEL();

    // d_weights->reshape({N, weights->getSize()});
    // d_weights->sumToDim_(1);    // (N, C_out, C_in) -> (C_out, C_in)
    // d_weights->reshape(weights->getShape());
    // gradients->sumToDim(d_bias, 1);
    // // accumulate grads
    // this->weights->acc_grads(d_weights);
    // this->bias->acc_grads(d_bias);

    if(input->getShape() != shape_in) {
        input->transpose(-2, -1);
    }

    size_t dim = gradients->getDim();

    DEBUG_PRINT("HERE4\n");

    if(dim == 3) {
        size_t C_in = in_channels, C_out = out_channels;
        size_t L = shape_in[2], N = shape_in[0];
        DimVector shape_o = {N, in_channels, L};
        this->d_in->reset(shape_o);

        dim3 block(BLOCK_SIZE3D, BLOCK_SIZE3D, BATCH_BASE);
        dim3 grid((L-1)/(BLOCK_SIZE3D*TN)+1, (C_out-1)/(BLOCK_SIZE3D*TM)+1, (N-1)/BATCH_BASE + 1);

        if(C_in % 4 == 0 && L % 4 == 0) {
            kConv1d_back_v3<<<grid, block>>>(gradients->getData(), d_in->getData(), weights->getData(), C_in, C_out, L, N); CHECK_KERNEL();
        } else {
            // DEBUG_PRINT("C_in=%d, C_out=%d, N=%d\n", C_in, C_out, N);
            kConv1d_back_v2<<<grid, block>>>(gradients->getData(), d_in->getData(), weights->getData(), C_in, C_out, L, N); CHECK_KERNEL();
        }
    } else {
        ERROR("Dimension not allowed!\n");
    }

    return this->d_in;
}
