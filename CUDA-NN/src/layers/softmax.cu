#include "softmax.cuh"

__global__
void kSoftmax(float* d_data, float* d_out, size_t C, size_t L, bool apply_log) {
    // It'll be faster if blocksize is the factor of L.
    int x = blockIdx.x;
    int tid = threadIdx.x;

    if(x >= C) return;

    __shared__ float sd_data[BLOCK_SIZE1D];
    __shared__ float sd_M[BLOCK_SIZE1D];
    float cur_max = 0.0f;
    float sum = 0.0f;

    // reduce to get maximum `cur_max`
    int iter = (L-1)/BLOCK_SIZE1D + 1;
    for(int i=0; i<iter; i++) {
        if (i*BLOCK_SIZE1D + tid < L) {
            sd_data[tid] = d_data[x*L + i*BLOCK_SIZE1D + tid];
        }
        __syncthreads();
        if (i*BLOCK_SIZE1D + tid < L) {
            sd_M[tid] = sd_data[tid];
        }
        __syncthreads();

        for(int stride=blockDim.x/2; stride>0; stride>>=1) {
            if(tid < stride && tid + stride + i*BLOCK_SIZE1D < L) {
                sd_M[tid] = sd_M[tid] > sd_M[tid + stride] ? sd_M[tid] : sd_M[tid+stride];
            }
            __syncthreads();
        }
        cur_max = cur_max >= sd_M[0] ? cur_max : sd_M[0];
    }

    for(int i=0; i<iter; i++) {
        if (i*BLOCK_SIZE1D + tid < L) {
            sd_data[tid] = expf(sd_data[tid] - cur_max);
        }
        __syncthreads();

        if (i*BLOCK_SIZE1D + tid < L) {
            sd_M[tid] = sd_data[tid];
        }
        __syncthreads();

        for(int stride=blockDim.x/2; stride>0; stride>>=1) {
            if(tid < L && tid + stride < L && tid < stride && tid + stride + i*BLOCK_SIZE1D < L) {
                sd_M[tid] = sd_M[tid] + sd_M[tid + stride];
            }
            __syncthreads();
        }
        sum = sd_M[0];
    }

    for(int i=0; i<iter; i++) {
        if(tid < L) {
            if(apply_log) {
                d_out[x*L + i*BLOCK_SIZE1D + tid] = logf(sd_data[tid]/(sum + 1e-4));
            } else {
                d_out[x*L + i*BLOCK_SIZE1D + tid] = sd_data[tid]/(sum + 1e-4);
            }
        }
    }
}


__global__
void kSoftMaxBP(float* d_out, float* d_logits, float* d_grad, size_t N, size_t L) {
    // assume that not apply_log
    // It'll be faster if blocksize is the factor of L.
    int x = blockIdx.x;
    int tid = threadIdx.x;

    int iter = (L-1)/BLOCK_SIZE1D + 1;
    for(int i=0; i<iter; i++) {
        if(tid + i*BLOCK_SIZE1D < L && x < N) {
            float logits = d_logits[x*L + tid + i*BLOCK_SIZE1D];
            d_grad[x*L + tid + i*BLOCK_SIZE1D] = d_out[x*L + tid + i*BLOCK_SIZE1D] * logits * (1-logits);
        }
    }
}

SoftMax::SoftMax(std::string prefix, size_t dim, bool apply_log) {
    this->dim = dim;
    this->prefix = prefix;
    this->apply_log = apply_log;
}

SoftMax::SoftMax(size_t dim, bool apply_log) {
    this->dim = dim;
    this->apply_log = apply_log;
}

SoftMax* SoftMax::train() {
    this->is_training = true;
    this->apply_log = false;
    return this;
}

Tensor* SoftMax::forward(Tensor* data) {
    DEBUG_PRINT("[SoftMax] %sforward\n", this->prefix.c_str());

    DimVector shape_o = data->getShape();
    if(this->output == nullptr) {
        this->output = new Tensor(shape_o);
    } this->output->reset(shape_o);    

    if(this->is_training)
        this->input = data;
    //////////////////////////////////////
    // PS: Naive impl is as follow
    // Tensor* x_max = data->max(this->dim);
    // Tensor* z = data->sub(x_max);
    // Tensor* nominator = z->exp();
    // Tensor* denominator = nominator->sum(this->dim);
    // this->output = nominator->div(denominator);
    // this->output->log_();
    // delete x_max, z, nominator, denominator;
    if(data->getDim() != 2 || this->dim != 1) {
        ERROR("Not implemented!\n");
    }

    size_t L = data->getSize(1), N = data->getSize(0);

    int block = BLOCK_SIZE1D;
    int grid = N;

    kSoftmax<<<grid, block>>>(data->getData(), this->output->getData(), N, L, !this->is_training); CHECK_KERNEL();

    return this->output;
}

Tensor* SoftMax::backward(Tensor* gradients) {
    DEBUG_PRINT("[SoftMax] %sbackward\n", this->prefix.c_str());

    if(input->getDim() != 2 || this->dim != 1) {
        ERROR("Not implemented!\n");
    }

    DimVector shape_o = input->getShape();
    size_t N = shape_o[0], L = shape_o[1];
    if(this->d_in == nullptr) {
        this->d_in = new Tensor(shape_o);
    } this->d_in->reset(shape_o);

    int block = BLOCK_SIZE1D;
    int grid = N;
    kSoftMaxBP<<<grid, block>>>(gradients->getData(), output->getData(), this->d_in->getData(), N, L); CHECK_KERNEL();
    return this->d_in;
} 