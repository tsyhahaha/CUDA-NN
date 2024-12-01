#include "cross_entropy.cuh"
/*
logits: (N x L)
labels: (N x L) one-hot
*/
__global__
void kSoftMaxCrossEntropyLoss(float* logits, float* labels, float* loss, int N, int L) {
    // It'll be faster if blocksize is the factor of L.
    int x = blockIdx.x;
    int tid = threadIdx.x;

    if(x >= N) return;

    __shared__ float sd_data[BLOCK_SIZE1D];
    float cur_max = 0.0f;
    float sum = 0.0f;
    float tmpError = 0.0f;

    // reduce to get maximum `cur_max`
    int iter = (L-1)/BLOCK_SIZE1D + 1;
    for(int i=0; i<iter; i++) {
        if (i*BLOCK_SIZE1D + tid < L) {
            sd_data[tid] = logits[x*L + i*BLOCK_SIZE1D + tid];
        }
        __syncthreads();

        for(int stride=blockDim.x/2; stride>0; stride>>=1) {
            if(tid < stride && tid + stride + i*BLOCK_SIZE1D < L) {
                sd_data[tid] = sd_data[tid] > sd_data[tid + stride] ? sd_data[tid] : sd_data[tid+stride];
            }
            __syncthreads();
        }
        cur_max = cur_max >= sd_data[0] ? cur_max : sd_data[0];
    }

    // exp()
    for(int i=0; i<iter; i++) {
        if (i*BLOCK_SIZE1D + tid < L) {
            sd_data[tid] = expf(logits[x*L + i*BLOCK_SIZE1D + tid] - cur_max);
        }
        __syncthreads();

        for(int stride=blockDim.x/2; stride>0; stride>>=1) {
            if(tid < stride && tid + stride + i*BLOCK_SIZE1D < L) {
                sd_data[tid] = sd_data[tid] + sd_data[tid + stride];
            }
            __syncthreads();
        }
        sum += sd_data[0];
    }

    for(int i=0; i<iter; i++) {
        if (i*BLOCK_SIZE1D + tid < L) {
            sd_data[tid] = expf(logits[x*L + i*BLOCK_SIZE1D + tid] - cur_max);
        }
        __syncthreads();

        float sm_output = sd_data[tid]/sum;
        __syncthreads();

        float label = labels[x*L + i*BLOCK_SIZE1D + tid];
        if (i*BLOCK_SIZE1D + tid < L) {
            tmpError -=  label * logf(sm_output) + (1-label) * logf(1 - sm_output);
        }
    }
    atomicAdd(loss, tmpError);
}

CrossEntropyLoss::CrossEntropyLoss(std::string reduction) {
    this->reduction = reduction;
    CHECK(cudaMalloc((float**)&d_loss, sizeof(float)));
    h_loss = (float*)malloc(sizeof(float));
}
CrossEntropyLoss::~CrossEntropyLoss() {
    CHECK(cudaFree(&d_loss));
    free(h_loss);
}

float CrossEntropyLoss::forward(Tensor* logits, Tensor* labels) {
    size_t N = logits->getSize(0), L = logits->getSize(1);
    int block = BLOCK_SIZE1D;
    dim3 grid = N;

    CHECK(cudaMemset(d_loss, 0, sizeof(float)));
    kSoftMaxCrossEntropyLoss<<<grid, block>>>(logits->getData(), labels->getData(), d_loss, N, L); CHECK_KERNEL();
    CHECK(cudaMemcpy(h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost));

    return *h_loss;
}

Tensor* CrossEntropyLoss::backward(Tensor*& gradients) {
    return nullptr;
}