#include "negative_likelihood.cuh"

__global__ 
void kNegativeLikelihoodLoss(float* logits, float* labels, float* loss, bool mean,int N, int L) {
    int bidx = blockIdx.x;
    int tid = threadIdx.x;

    if(bidx >= N) return;

    __shared__ float sd_data[BLOCK_SIZE1D];
    float tmpError = 0.0f;

    int iter = (L-1)/BLOCK_SIZE1D + 1;
    for(int i=0; i<iter; i++) {
        if(i * BLOCK_SIZE1D + tid < L) {
            float label = labels[bidx*L + i*BLOCK_SIZE1D + tid];
            float pred = logits[bidx*L + i * BLOCK_SIZE1D + tid];
            sd_data[tid] = -label * logf(pred);
        }
        __syncthreads();

        for(int stride=blockDim.x/2; stride>0; stride>>=1) {
            if(tid < stride && tid + stride + i*BLOCK_SIZE1D < L) {
                sd_data[tid] = sd_data[tid] + sd_data[tid + stride];
            }
            __syncthreads();
        }

        if(tid == 0) {
            if(mean) {
                tmpError += sd_data[0] / L;
            } else {
                tmpError += sd_data[0];
            }
        }
    }
    if(tid == 0) {
        atomicAdd(loss, tmpError);
    }
}

__global__ 
void kNegativeLikelihoodLossBP(float* logits, float* labels, float* d_out, bool mean, int N, int L) {
    int bidx = blockIdx.x;
    int tid = threadIdx.x;

    if(bidx >= N) return;

    int iter = (L-1)/BLOCK_SIZE1D + 1;
    for(int i=0; i<iter; i++) {
        if(i * BLOCK_SIZE1D + tid < L) {
            float label = labels[bidx*L + i*BLOCK_SIZE1D + tid];
            float pred = logits[bidx*L + i * BLOCK_SIZE1D + tid];
            
            if(mean) {
                d_out[bidx*L + i * BLOCK_SIZE1D + tid] = -(label / pred) / L;
            } else {
                d_out[bidx*L + i * BLOCK_SIZE1D + tid] = - label / pred;
            }
        }
    }
}

NegativeLikelihoodLoss::NegativeLikelihoodLoss(std::string reduction) {
    this->reduction = reduction;
    CHECK(cudaMalloc((float**)&d_loss, sizeof(float)));
    h_loss = (float*)malloc(sizeof(float));
    CHECK(cudaStreamCreate(&s));
}

NegativeLikelihoodLoss::~NegativeLikelihoodLoss() {
    CHECK(cudaFree(&d_loss));
    free(h_loss);

    if(!this->logits)
        delete logits;
    if(!this->labels)
        delete labels;
    CHECK(cudaStreamDestroy(s));
}

float NegativeLikelihoodLoss::forward(Tensor* logits, Tensor* labels) {
    DEBUG_PRINT("[NegativeLikelihoodLoss] forward\n");
    CHECK(cudaMemset(d_loss, 0, sizeof(float)));

    size_t N = logits->getSize(0), L = logits->getSize(1);
    int block = BLOCK_SIZE1D;
    dim3 grid = N;

    kNegativeLikelihoodLoss<<<grid, block>>>(logits->getData(), labels->getData(), d_loss, this->reduction == "mean", N, L); CHECK_KERNEL();

    this->logits = logits;
    this->labels = labels;

    CHECK(cudaMemcpy(h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost));
    return *h_loss;
}

Tensor* NegativeLikelihoodLoss::backward(Tensor*& gradients) {
    DEBUG_PRINT("[NegativeLikelihoodLoss] backward\n");
    size_t N = logits->getSize(0), L = logits->getSize(1);

    DimVector shape_o = {N, L};
    if(gradients == nullptr) {
        gradients = new Tensor(shape_o);
    } gradients->reset(shape_o);

    int block = BLOCK_SIZE1D;
    dim3 grid = N;

    kNegativeLikelihoodLossBP<<<grid, block>>>(logits->getData(), labels->getData(), gradients->getData(), reduction == "mean", N, L); CHECK_KERNEL();

    return gradients;
}