#include "tensor.cuh"
#include "conv1d.cuh"

__global__
void kConv1d(float* d_in, float* d_out, float* weights, float* bias, int C_in, int C_out, int L, int N) {
    // weights(C_out x C_in) @ d_in(B x C_in x L) + bias(C_out)= (B x C_out x L)
    float cVal = 0.0f;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float ds_A[BLOCK_SIZE2D][TILE_SIZE];
    __shared__ float ds_B[TILE_SIZE][BLOCK_SIZE2D];
    __shared__ float ds_bias[BLOCK_SIZE2D];

    if(threadIdx.x==0)
        ds_bias[threadIdx.y] = bias[row];

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
            d_out[b*C_out*L + row*L + col] = cVal + ds_bias[threadIdx.y];
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
    DEBUG_PRINT("in the constructor\n");
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

        dim3 block(BLOCK_SIZE2D, BLOCK_SIZE2D);
        dim3 grid((L-1)/BLOCK_SIZE2D+1, (C_out-1)/BLOCK_SIZE2D+1);

        kConv1d<<<grid, block>>>(data->getData(), output->getData(), weights->getData(), bias->getData(), C_in, C_out, L, N); CHECK_KERNEL();
    } else {
        ERROR("Dimension not allowed!\n");
    }

    return this->output;
}

Tensor* Conv1d::backward(Tensor* gradients) {
    return nullptr;
}
