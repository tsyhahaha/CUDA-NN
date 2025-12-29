#include "softmax.cuh"

__global__
void kSoftmax(float* d_data, float* d_out, size_t C, size_t L, bool apply_log) {
    // It'll be faster if blocksize is the factor of L.
    int x = blockIdx.x;
    int tid = threadIdx.x;

    __shared__ float sd_M[BLOCK_SIZE1D];
    float cur_max = -1e30f;
    float sum = 0.0f;

    // Step 1: reduce to get maximum `cur_max`
    int iter = (L-1)/BLOCK_SIZE1D + 1;
    for(int i=0; i<iter; i++) {
        int idx = i*BLOCK_SIZE1D + tid;
        if (idx < L) {
            sd_M[tid] = d_data[x*L + idx];
        } else {
            sd_M[tid] = -1e30f;
        }
        __syncthreads();

        for(int stride=blockDim.x/2; stride>0; stride>>=1) {
            if(tid < stride) {
                sd_M[tid] = sd_M[tid] > sd_M[tid + stride] ? sd_M[tid] : sd_M[tid + stride];
            }
            __syncthreads();
        }
        cur_max = cur_max >= sd_M[0] ? cur_max : sd_M[0];
        __syncthreads();
    }

    // Step 2: compute exp(x - max) and reduce to get sum
    for(int i=0; i<iter; i++) {
        int idx = i*BLOCK_SIZE1D + tid;
        if (idx < L) {
            sd_M[tid] = expf(d_data[x*L + idx] - cur_max);
        } else {
            sd_M[tid] = 0.0f;
        }
        __syncthreads();

        for(int stride=blockDim.x/2; stride>0; stride>>=1) {
            if(tid < stride) {
                sd_M[tid] = sd_M[tid] + sd_M[tid + stride];
            }
            __syncthreads();
        }
        sum = sum + sd_M[0];
        __syncthreads();
    }

    // Step 3: normalization and write output
    for(int i=0; i<iter; i++) {
        int idx = i*BLOCK_SIZE1D + tid;
        if (idx < L) {
            float exp_val = expf(d_data[x*L + idx] - cur_max);
            if(apply_log) {
                d_out[x*L + idx] = logf(exp_val / sum);
            } else {
                d_out[x*L + idx] = exp_val / sum;
            }
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

Tensor* SoftMax::forward(Tensor* data) {

    DimVector shape_o = data->getShape();
    if(this->output == nullptr) {
        this->output = new Tensor(shape_o);
    }
    this->output->reset(shape_o);    
    
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

    size_t L = data->getShape()[1], C = data->getShape()[0];

    if(this->output == nullptr) {
        this->output = new Tensor({C, L});
    }

    int block = BLOCK_SIZE1D;
    int grid = C;
    kSoftmax<<<grid, block>>>(data->getData(), this->output->getData(), C, L, this->apply_log); CHECK_KERNEL();

    return this->output;
}

Tensor* SoftMax::backward(Tensor* gradients) {
    return nullptr;
} 