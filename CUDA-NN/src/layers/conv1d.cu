#include "tensor.cuh"
#include "conv1d.cuh"

#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])

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

    int offset_in  = batch * C_in  * L;
    int offset_out = batch * C_out * L;

    // batch b
    float cVal = 0.0f;
    int phase = (C_in - 1) / TILE_SIZE + 1;
    for(int p=0; p<phase;p++) {
        if (row < C_out && threadIdx.x < TILE_SIZE && threadIdx.z == 0) {
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
            cVal += ds_A[threadIdx.y][i] * ds_B[threadIdx.z][i][threadIdx.x];   // no bank conflict when reading from ds_A --- TILE_SIZE * BLOCK_SIZE3D stridd
        }
        __syncthreads();
    }
    if(row < C_out && col < L)
        d_out[offset_out + row*L + col] = cVal + bias[row];
}

__global__
void kConv1d_v2(float* d_in, float* d_out, float* weights, float* bias, int C_in, int C_out, int L, int N) {
    // weights(C_out x C_in) @ d_in(B x C_in x L) + bias(C_out)= (B x C_out x L)
    int row = (blockIdx.y * blockDim.y + threadIdx.y) * TM;
    int col = (blockIdx.x * blockDim.x + threadIdx.x) * TN;
    int batch = blockIdx.z * blockDim.z + threadIdx.z;

    if(batch >= N) return;

    __shared__ float ds_A[BLOCK_SIZE3D * TM][TILE_SIZE];
    __shared__ float ds_B[BATCH_BASE][TILE_SIZE][BLOCK_SIZE3D * TN];
    float cVal[TM][TN] = {0.0f};

    // batch
    int offset1 = batch*C_in*L;
    int offset2 = batch*C_out*L;

    int phase = (C_in - 1) / TILE_SIZE + 1;
    for(int p=0; p<phase;p++) {
        if(threadIdx.x < TILE_SIZE && threadIdx.z == 0) {
            for(int i=0; i<TM; i++) {
                if((row + i) < C_out && p*TILE_SIZE + threadIdx.x < C_in) {
                    // ds_A[batch][threadIdx.y * TM + i][threadIdx.x]
                    ds_A[threadIdx.y*TM+i][threadIdx.x] = weights[(row + i)*C_in + p*TILE_SIZE + threadIdx.x];
                } else {
                    ds_A[threadIdx.y*TM+i][threadIdx.x] = 0.0f;
                }
            }
        }

        if(threadIdx.y < TILE_SIZE){
            for(int j=0; j<TN; j++) {
                if((col + j)< L && p*TILE_SIZE + threadIdx.y < C_in) {
                    // ds_B[batch][threadIdx.y][threadIdx.x * TN + j]
                    ds_B[threadIdx.z][threadIdx.y][threadIdx.x*TN+j] = d_in[offset1 + (p*TILE_SIZE + threadIdx.y)*L + col + j];
                } else {
                    ds_B[threadIdx.z][threadIdx.y][threadIdx.x*TN+j] = 0.0f;
                }      
            }
        }
        __syncthreads();

        for (int i=0; i<TM; i++) {
            for(int j = 0; j<TN; j++) {
                for(int k = 0; k<TILE_SIZE; k++) {
                    // ds_A[batch][threadIdx.y * TM + i][k]
                    // ds_B[batch][k][threadIdx.x * TN + j]
                    cVal[i][j] += ds_A[threadIdx.y * TM + i][k] * ds_B[threadIdx.z][k][threadIdx.x * TN + j];
                }
                __syncthreads();
            }
        }
        
    }

    for(int i=0; i<TM; i++) {
        for (int j=0; j<TN; j++) {
            if(row+i < C_out && col+j < L)
                d_out[offset2 + (row + i)*L + col + j] = cVal[i][j] + bias[row+i];
        }
    }
}

__global__
void kConv1d_v3(float* d_in, float* d_out, float* weights, float* bias, int C_in, int C_out, int L, int N) {
    // weights(C_out x C_in) @ d_in(B x C_in x L) + bias(C_out)= (B x C_out x L)
    int row = (blockIdx.y * blockDim.y + threadIdx.y) * TM;
    int col = (blockIdx.x * blockDim.x + threadIdx.x) * TN;
    int batch = blockIdx.z * blockDim.z + threadIdx.z;

    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    if(batch >= N) return;

    __shared__ float ds_A[BLOCK_SIZE3D * TM][TILE_SIZE];
    __shared__ float ds_B[BATCH_BASE][TILE_SIZE][BLOCK_SIZE3D * TN];
    float cVal[TM][TN] = {0.0f};

    // batch
    int offset1 = batch*C_in*L;
    int offset2 = batch*C_out*L;

    int ds_A_row = tid / 2;
    int ds_A_col = tid % 2 * 4;
    int ds_B_row = tid / 32;
    int ds_B_col = tid % 32 * 4;

    int phase = (C_in - 1) / TILE_SIZE + 1;
    for(int p=0; p<phase;p++) {
        // thread to read ds_A
        
        if(threadIdx.z == 0 && row + ds_A_row + 3 < C_out && p*TILE_SIZE + ds_A_col + 3 < C_in && ds_A_row + 4 < BLOCK_SIZE3D * TM && ds_A_col + 4 < TILE_SIZE)
            FETCH_FLOAT4(ds_A[ds_A_row][ds_A_col]) = FETCH_FLOAT4(weights[(row + ds_A_row)*C_in + p*TILE_SIZE + ds_A_col]);

        // if(threadIdx.x < TILE_SIZE && threadIdx.z == 0) {
        //     for(int i=0; i<TM/4; i++) {
        //         if((row + i) < C_out && p*TILE_SIZE + threadIdx.x < C_in) {
        //             // ds_A[batch][threadIdx.y * TM + i][threadIdx.x]
        //             ds_A[threadIdx.y*TM+i][threadIdx.x] = weights[(row + i)*C_in + p*TILE_SIZE + threadIdx.x];
        //         } else {
        //             ds_A[threadIdx.y*TM+i][threadIdx.x] = 0.0f;
        //         }
        //     }
        // }

        if(p*TILE_SIZE + ds_B_row + 4 < C_in && ds_B_col + 4 < L && ds_B_row + 4 < TILE_SIZE && ds_B_col + 4 < BLOCK_SIZE3D * TN)
        FETCH_FLOAT4(ds_B[blockIdx.z][ds_B_row][ds_B_col]) = FETCH_FLOAT4(d_in[offset1 + (p*TILE_SIZE + ds_B_row) * L + ds_B_col]);

        // if(threadIdx.y < TILE_SIZE){
        //     for(int j=0; j<TN/4; j++) {
        //         if((col + j)< L && p*TILE_SIZE + threadIdx.y < C_in) {
        //             // ds_B[batch][threadIdx.y][threadIdx.x * TN + j]
        //             ds_B[threadIdx.z][threadIdx.y][threadIdx.x*TN+j] = d_in[offset1 + (p*TILE_SIZE + threadIdx.y)*L + col + j];
        //         } else {
        //             ds_B[threadIdx.z][threadIdx.y][threadIdx.x*TN+j] = 0.0f;
        //         }      
        //     }
        // }
        __syncthreads();

        for (int i=0; i<TM; i++) {
            for(int j = 0; j<TN; j++) {
                for(int k = 0; k<TILE_SIZE; k++) {
                    // ds_A[batch][threadIdx.y * TM + i][k]
                    // ds_B[batch][k][threadIdx.x * TN + j]
                    cVal[i][j] += ds_A[threadIdx.y * TM + i][k] * ds_B[threadIdx.z][k][threadIdx.x * TN + j];
                }
                __syncthreads();
            }
        }
        
    }

    for(int i=0; i<TM; i++) {
        for (int j=0; j<TN; j++) {
            if(row+i < C_out && col+j < L)
                d_out[offset2 + (row + i)*L + col + j] = cVal[i][j] + bias[row+i];
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
        this->bias = new Tensor({out_channels}, ZERO);
    } else 
        this->bias = nullptr;

    this->weights = new Tensor({out_channels, in_channels}, RANDOM);
}



Conv1d::Conv1d(size_t in_channels, size_t out_channels, size_t kernel_size, bool bias) {
    printf("in the constructor\n");
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

Conv1d::~Conv1d() {
    delete weights;
    delete bias;
    delete input;
    delete output;
    delete outputBackward;
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
        dim3 grid((L-1)/(BLOCK_SIZE3D*TN)+1, (C_out-1)/(BLOCK_SIZE3D*TM)+1, (N-1)/BATCH_BASE+1);

        // if(L % 8 == 0 && C_in % 8 == 0 && C_out % 8 == 0) {
        //     printf("HERE!\n");
        //     kConv1d_v3<<<grid, block>>>(data->getData(), output->getData(), weights->getData(), bias->getData(), C_in, C_out, L, N); CHECK_KERNEL();
        // } else {
        kConv1d_v2<<<grid, block>>>(data->getData(), output->getData(), weights->getData(), bias->getData(), C_in, C_out, L, N); CHECK_KERNEL();
        // }
    } else {
        ERROR("Dimension not allowed!\n");
    }

    return this->output;
}

Tensor* Conv1d::backward(Tensor* gradients) {
    return nullptr;
}
