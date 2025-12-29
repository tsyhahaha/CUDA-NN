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
    float cur_max = -1e30f;
    float sum = 0.0f;
    float tmpError = 0.0f;

    // Step 1: reduce to get maximum `cur_max`
    int iter = (L-1)/BLOCK_SIZE1D + 1;
    for(int i=0; i<iter; i++) {
        int idx = i*BLOCK_SIZE1D + tid;
        if (idx < L) {
            sd_data[tid] = logits[x*L + idx];
        } else {
            sd_data[tid] = -1e30f;
        }
        __syncthreads();

        for(int stride=blockDim.x/2; stride>0; stride>>=1) {
            if(tid < stride) {
                sd_data[tid] = sd_data[tid] > sd_data[tid + stride] ? sd_data[tid] : sd_data[tid+stride];
            }
            __syncthreads();
        }
        cur_max = cur_max >= sd_data[0] ? cur_max : sd_data[0];
        __syncthreads();
    }

    // Step 2: compute exp(x - max) and sum
    for(int i=0; i<iter; i++) {
        int idx = i*BLOCK_SIZE1D + tid;
        if (idx < L) {
            sd_data[tid] = expf(logits[x*L + idx] - cur_max);
        } else {
            sd_data[tid] = 0.0f;
        }
        __syncthreads();

        for(int stride=blockDim.x/2; stride>0; stride>>=1) {
            if(tid < stride) {
                sd_data[tid] = sd_data[tid] + sd_data[tid + stride];
            }
            __syncthreads();
        }
        sum += sd_data[0];
        __syncthreads();
    }

    // Step 3: compute cross entropy loss
    for(int i=0; i<iter; i++) {
        int idx = i*BLOCK_SIZE1D + tid;
        if (idx < L) {
            float exp_val = expf(logits[x*L + idx] - cur_max);
            float sm_output = fmaxf(fminf(exp_val / (sum + 1e-8f), 1.0f - 1e-8f), 1e-8f);
            float label = labels[x*L + idx];
            tmpError -= label * logf(sm_output) + (1.0f - label) * logf(1.0f - sm_output);
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