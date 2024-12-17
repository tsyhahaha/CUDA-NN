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
    int batch = blockIdx.z;

    __shared__ float ds_A[BLOCK_SIZE3D][TILE_SIZE];
    __shared__ float ds_B[TILE_SIZE][BLOCK_SIZE3D];

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
                ds_B[threadIdx.y][threadIdx.x] = d_in[offset_in + (p*TILE_SIZE + threadIdx.y)*L + col];
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
        d_out[offset_out + row*L + col] = cVal + bias[row];
}

__global__
void kConv1d_v2(float* d_in, float* d_out, float* weights, float* bias, int C_in, int C_out, int L, int N) {
    // weights(C_out x C_in) @ d_in(B x C_in x L) + bias(C_out)= (B x C_out x L)
    int row = (blockIdx.y * blockDim.y + threadIdx.y)*TM;
    int col = (blockIdx.x * blockDim.x + threadIdx.x)*TN;
    int batch = blockIdx.z;

    __shared__ float ds_A[BLOCK_SIZE3D * TM][TILE_SIZE];
    __shared__ float ds_B[TILE_SIZE][BLOCK_SIZE3D * TN];

    float tie[TM][TN] = {0.0f};
    float reg_A[TM] = {0.0f};
    float reg_B[TN] = {0.0f};

    uint offset_in = batch * C_in * L;
    uint offset_out = batch * C_out * L;

    if(batch >= N) return;

    int phase = (C_in - 1) / TILE_SIZE + 1;
    for(int p=0; p<phase;p++) {
        for(int i=0; i<TM; i++) {
            if(threadIdx.x < TILE_SIZE) {
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
                    ds_B[threadIdx.y][threadIdx.x*TN + j] = d_in[offset_in + (p*TILE_SIZE + threadIdx.y)*L + col + j];
                } else {
                    ds_B[threadIdx.y][threadIdx.x*TN + j] = 0.0f;
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
                    reg_B[j] = ds_B[k][threadIdx.x * TN + j];
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
    int batch = blockIdx.z;

    if(batch >= N) return;

    __shared__ float ds_w[TILE_SIZE][BLOCK_SIZE3D * TM];
    __shared__ float ds_in[TILE_SIZE][BLOCK_SIZE3D * TN];

    float tie[TM][TN] = {0.0f};
    float reg_w[TM] = {0.0f};
    float reg_in[TN] = {0.0f};
    float tmp[4] = {0.0f};

    uint offset_in = batch * C_in * L;
    uint offset_out = batch * C_out * L;

    int phase = (C_in - 1) / TILE_SIZE + 1;
    for(int p=0; p<phase;p++) {
        for(int i=0; i<TM; i++) {
            if(threadIdx.x*4 < TILE_SIZE) {
                if(row+i < C_out && p*TILE_SIZE + threadIdx.x*4 < C_in) {
                    FETCH_FLOAT4(tmp[0]) = FETCH_FLOAT4(weights[(row+i) * C_in + p*TILE_SIZE + threadIdx.x*4]);

                    ds_w[threadIdx.x*4+0][threadIdx.y*TM+i] = tmp[0];
                    ds_w[threadIdx.x*4+1][threadIdx.y*TM+i] = tmp[1];
                    ds_w[threadIdx.x*4+2][threadIdx.y*TM+i] = tmp[2];
                    ds_w[threadIdx.x*4+3][threadIdx.y*TM+i] = tmp[3];
                } else {
                    ds_w[threadIdx.x*4+0][threadIdx.y*TM+i] = 0.0f;
                    ds_w[threadIdx.x*4+1][threadIdx.y*TM+i] = 0.0f;
                    ds_w[threadIdx.x*4+2][threadIdx.y*TM+i] = 0.0f;
                    ds_w[threadIdx.x*4+3][threadIdx.y*TM+i] = 0.0f;
                }
            }
        }
        

        for(int j=0; j<TN/4; j++) {
            if(threadIdx.y < TILE_SIZE) {
                if (col + j*4 < L && p*TILE_SIZE + threadIdx.y < C_in) {
                    FETCH_FLOAT4(ds_in[threadIdx.y][threadIdx.x*TN + j*4]) = FETCH_FLOAT4(d_in[offset_in + (p*TILE_SIZE + threadIdx.y)*L + col + j*4]);
                } else {
                    ds_in[threadIdx.y][threadIdx.x*TN + j*4 + 0] = 0.0f;
                    ds_in[threadIdx.y][threadIdx.x*TN + j*4 + 1] = 0.0f;
                    ds_in[threadIdx.y][threadIdx.x*TN + j*4 + 2] = 0.0f;
                    ds_in[threadIdx.y][threadIdx.x*TN + j*4 + 3] = 0.0f;
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
                    FETCH_FLOAT4(reg_in[j*4]) = FETCH_FLOAT4(ds_in[k][threadIdx.x * TN + j*4]);
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
    int row = (blockIdx.y * blockDim.y + threadIdx.y)*TM;
    int col = (blockIdx.x * blockDim.x + threadIdx.x)*TN;
    int batch = blockIdx.z;

    __shared__ float ds_A[BLOCK_SIZE3D * TM][TILE_SIZE];
    __shared__ float ds_B[TILE_SIZE][BLOCK_SIZE3D * TN];

    float tie[TM][TN] = {0.0f};
    float reg_A[TM] = {0.0f};
    float reg_B[TN] = {0.0f};

    if(batch >= N) return;

    uint offset_in = batch * C_out * L;
    uint offset_out = batch * C_in * L;

    int phase = (C_out - 1) / TILE_SIZE + 1;
    for(int p=0; p<phase;p++) {
        for(int i=0; i<TM; i++) {
            if(threadIdx.x < TILE_SIZE) {
                if(row + i < C_in && p*TILE_SIZE + threadIdx.x < C_out) {
                    // weights[C_out][C_in]
                    ds_A[threadIdx.y*TM+i][threadIdx.x] = weights[(p*TILE_SIZE + threadIdx.x) * C_in + row + i];
                } else {
                    ds_A[threadIdx.y*TM+i][threadIdx.x] = 0.0f;
                }
            }
        }

        for(int j=0; j<TN; j++) {
            if(threadIdx.y < TILE_SIZE) {
                if (col + j < L && p*TILE_SIZE + threadIdx.y < C_out) {
                    ds_B[threadIdx.y][threadIdx.x*TN + j] = gradients[offset_in + (p*TILE_SIZE + threadIdx.y)*L + col + j];
                } else {
                    ds_B[threadIdx.y][threadIdx.x*TN + j] = 0.0f;
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
                    reg_B[j] = ds_B[k][threadIdx.x * TN + j];
                } else {
                    reg_B[j] = 0.0f;
                }
            }
            __syncthreads();

            for(int j=0; j<TN; j++) {
                for (int i=0; i<TM; i++) {
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
void kBackprop_to_bias(
    float* gradients, float* d_bias, size_t N, size_t C_out, size_t L
) {
    float d_bias_acc = 0.0f;
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int batch = blockIdx.z;
    int tid = threadIdx.x;

    __shared__ float sd_M[BATCH_BASE][BLOCK_SIZE1D];

    if(batch >= N || row >= C_out) return;

    int offset = batch * C_out * L;
    int phase = (L-1)/BLOCK_SIZE1D + 1;
    for(int p=0; p<phase; p++) {
        int col = tid + p * BLOCK_SIZE1D;
        if(col < L) {
            sd_M[threadIdx.y][tid] = gradients[offset + row * L + col];
        } else {
            sd_M[threadIdx.y][tid] = 0.0f;
        }
        __syncthreads();

        for(int stride=BLOCK_SIZE1D/2; stride>0; stride>>=1) {
            if(tid < stride && tid + stride + p*BLOCK_SIZE1D < L) {
                sd_M[threadIdx.y][tid] += sd_M[threadIdx.y][tid + stride];
            }
            __syncthreads();
        }

        if(tid==0) {
            d_bias_acc += sd_M[threadIdx.y][0];
        }
    }

    if(tid == 0) {
        atomicAdd(&d_bias[row], d_bias_acc);
    }

}

__global__
void kBackprop_to_weights(
    float* gradients, float* input, float* d_weights, size_t N, size_t C_out, size_t C_in, size_t L
) {
    // gradients(N x C_out x L) @ input(N x C_in x L) = C (B x C_out x C_in)
    float cVal = 0.0f;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int batch = blockIdx.z;

    __shared__ float ds_A[BLOCK_SIZE2D][TILE_SIZE];
    __shared__ float ds_B[BLOCK_SIZE2D][TILE_SIZE];

    if(batch >= N) return;

    int offset1 = batch * C_out * L;
    int offset2 = batch * L * C_in;
    int phase = (L - 1) / TILE_SIZE + 1;
    for(int p=0; p<phase; p++) {
        if (threadIdx.x < TILE_SIZE) {
            if(row < C_out && p*TILE_SIZE + threadIdx.x < L) {
                ds_A[threadIdx.y][threadIdx.x] = gradients[offset1 + row*L + p*TILE_SIZE + threadIdx.x];
            } else {
                ds_A[threadIdx.y][threadIdx.x] = 0.0f;
            }
        }

        if(threadIdx.y < TILE_SIZE){
            if(col < C_in && p*TILE_SIZE + threadIdx.y < L) {
                ds_B[threadIdx.x][threadIdx.y] = input[offset2 + col*L + (p*TILE_SIZE + threadIdx.y)];
            } else {
                ds_B[threadIdx.x][threadIdx.y] = 0.0f;
            }
        }
        __syncthreads();

        for (int i=0; i<TILE_SIZE; i++) {
            cVal += ds_A[threadIdx.y][i] * ds_B[threadIdx.x][i];
        }
        __syncthreads();

    }

    if (row < C_out && col < C_in) {
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
        size_t C_in = in_channels, C_out = out_channels, L = Configurer::cropping_size;
        dim3 block(BLOCK_SIZE2D, BLOCK_SIZE2D);
        dim3 grid((L-1)/BLOCK_SIZE2D+1, (C_out-1)/BLOCK_SIZE2D+1);

        kConv1d<<<grid, block>>>(data->getData(), output->getData(), weights->getData(), bias->getData(), C_in, C_out, L, 1); CHECK_KERNEL();
    } else if(dim == 3) {
        size_t C_in = in_channels, C_out = out_channels;
        size_t L = shape_in[2], N = shape_in[0];

        dim3 block(BLOCK_SIZE3D, BLOCK_SIZE3D);
        dim3 grid((L-1)/(BLOCK_SIZE3D*TN)+1, (C_out-1)/(BLOCK_SIZE3D*TM)+1, N);

        if(C_in % 4 == 0 && L % 4 == 0) {
            kConv1d_v3<<<grid, block>>>(data->getData(), output->getData(), weights->getData(), bias->getData(), C_in, C_out, L, N); CHECK_KERNEL();
        } else {
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

    Tensor* d_weights = weights->getGradsAcc();
    Tensor* d_bias = bias->getGradsAcc();

    dim3 block2(BLOCK_SIZE1D, BATCH_BASE);
    dim3 grid2(1, (out_channels-1)/BATCH_BASE + 1, N);
    kBackprop_to_bias<<<grid2, block2>>>(gradients->getData(), d_bias->getData(), N, out_channels, L); CHECK_KERNEL();

    dim3 block1(BLOCK_SIZE2D, BLOCK_SIZE2D);
    dim3 grid1((in_channels-1)/BLOCK_SIZE2D+1, (out_channels-1)/BLOCK_SIZE2D+1, N);
    kBackprop_to_weights<<<grid1, block1>>>(gradients->getData(), input->getData(), d_weights->getData(), N, out_channels, in_channels, L); CHECK_KERNEL();

    size_t dim = gradients->getDim();

    if(dim == 3) {
        size_t C_in = in_channels, C_out = out_channels;
        size_t L = Configurer::cropping_size, N = Configurer::batch_size;
        DimVector shape_o = {N, in_channels, L};
        this->d_in->reset(shape_o);

        dim3 block(BLOCK_SIZE3D, BLOCK_SIZE3D);
        dim3 grid((L-1)/(BLOCK_SIZE3D*TN)+1, (in_channels-1)/(BLOCK_SIZE3D*TM)+1, N);
        
        kConv1d_back_v2<<<grid, block>>>(gradients->getData(), d_in->getData(), weights->getData(), C_in, C_out, L, N); CHECK_KERNEL();
    } else {
        ERROR("Dimension not allowed!\n");
    }

    // Save for checking grads
    if(Configurer::track_grads && (Configurer::target == "conv" ||  Configurer::target == "all")) {
        save_vector_to_txt("/home/tsyhahaha/CUDA-NN/data/grads/" + this->prefix + "d_out.txt", gradients->toVec());
        save_vector_to_txt("/home/tsyhahaha/CUDA-NN/data/grads/" + this->prefix + "in.txt", input->toVec());
        save_vector_to_txt("/home/tsyhahaha/CUDA-NN/data/grads/" + this->prefix + "weights.txt", weights->toVec());
        save_vector_to_txt("/home/tsyhahaha/CUDA-NN/data/grads/" + this->prefix + "d_in.txt", d_in->toVec());
        save_vector_to_txt("/home/tsyhahaha/CUDA-NN/data/grads/" + this->prefix + "d_weights.txt", d_weights->toVec());
        save_vector_to_txt("/home/tsyhahaha/CUDA-NN/data/grads/" + this->prefix + "d_bias.txt", d_bias->toVec());
    }

    return this->d_in;
}
