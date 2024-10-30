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



Conv1d::Conv1d(std::string prefix, size_t in_channels, size_t out_channels, size_t kernel_size, bool bias) {
    this->in_channels = in_channels;
    this->out_channels = out_channels;
    this->kernel_size = this->kernel_size;

    this->prefix = prefix;

    if(kernel_size != 1) {
        perror("Not implemented!");
    }

    if(bias) {
        this->bias = new Tensor({out_channels}, this->is_training ? ZERO : NONE);
    } else 
        this->bias = nullptr;

    this->weights = new Tensor({out_channels, in_channels}, this->is_training ? RANDOM : NONE);
}


Conv1d::Conv1d(size_t in_channels, size_t out_channels, size_t kernel_size, bool bias) {
    this->in_channels = in_channels;
    this->out_channels = out_channels;
    this->kernel_size = this->kernel_size;

    if(kernel_size != 1) {
        perror("Not implemented!");
    }

    if(bias) {
        this->bias = new Tensor({out_channels}, ZERO);
    } else 
        this->bias = nullptr;

    this->weights = new Tensor({out_channels, in_channels}, RANDOM);
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
    this->reset();
    if(this->is_training)
        this->input = data;

    size_t dim = data->getDim();
    DimVector shape_in = data->getShape();
    if(dim == 2) {
        size_t C_in = in_channels, C_out = out_channels, L = shape_in[1];
        this->output = new Tensor({C_out, L});

        dim3 block(BLOCK_SIZE2D, BLOCK_SIZE2D);
        dim3 grid((L-1)/BLOCK_SIZE2D+1, (C_out-1)/BLOCK_SIZE2D+1);

        kConv1d<<<grid, block>>>(data->getData(), output->getData(), weights->getData(), bias->getData(), C_in, C_out, L, 1); CHECK_KERNEL();
    } else if(dim == 3) {
        size_t C_in = in_channels, C_out = out_channels;
        size_t L = shape_in[2], N = shape_in[0];
        this->output = new Tensor({N, C_out, L});

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
    return nullptr;
}
