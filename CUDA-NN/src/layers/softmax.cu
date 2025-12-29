#include "softmax.cuh"

__global__
void kSoftmax(float* d_data, float* d_out, size_t C, size_t L, bool apply_log) {
    // It'll be faster if blocksize is the factor of L.
    int x = blockIdx.x;
    int tid = threadIdx.x;

    if(x >= C) return;

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
        sum += sd_M[0];
        __syncthreads();
    }

    // Step 3: normalization and write output
    for(int i=0; i<iter; i++) {
        int idx = i*BLOCK_SIZE1D + tid;
        if (idx < L) {
            float exp_val = expf(d_data[x*L + idx] - cur_max);
            if(apply_log) {
                d_out[x*L + idx] = logf(exp_val / (sum + 1e-8f));
            } else {
                d_out[x*L + idx] = exp_val / (sum + 1e-8f);
            }
        }
    }
}


__global__
void kSoftMaxBP(float* d_out, float* softmax_out, float* d_grad, size_t N, size_t L) {
    // Softmax backward: d_input = softmax * (d_out - sum(d_out * softmax))
    // For each sample x, compute: d_grad[j] = softmax[j] * (d_out[j] - sum_k(d_out[k] * softmax[k]))
    int x = blockIdx.x;
    int tid = threadIdx.x;

    if(x >= N) return;

    __shared__ float sd_sum[BLOCK_SIZE1D];
    float local_sum = 0.0f;

    // Step 1: compute sum(d_out * softmax) for this sample
    int iter = (L-1)/BLOCK_SIZE1D + 1;
    for(int i=0; i<iter; i++) {
        int idx = i*BLOCK_SIZE1D + tid;
        if(idx < L) {
            sd_sum[tid] = d_out[x*L + idx] * softmax_out[x*L + idx];
        } else {
            sd_sum[tid] = 0.0f;
        }
        __syncthreads();

        for(int stride=blockDim.x/2; stride>0; stride>>=1) {
            if(tid < stride) {
                sd_sum[tid] += sd_sum[tid + stride];
            }
            __syncthreads();
        }
        local_sum += sd_sum[0];
        __syncthreads();
    }

    // Step 2: compute d_grad = softmax * (d_out - sum)
    for(int i=0; i<iter; i++) {
        int idx = i*BLOCK_SIZE1D + tid;
        if(idx < L) {
            float s = softmax_out[x*L + idx];
            d_grad[x*L + idx] = s * (d_out[x*L + idx] - local_sum);
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