#ifndef BASELOSS_CUH
#define BASELOSS_CUH

#include "tensor.cuh"
#include "utils.cuh"
#include "configure.cuh"
#include <string>

class BaseLoss {
    public:
        float* h_loss;
        float* d_loss;
        
    public:
        virtual float forward(Tensor* logits, Tensor* labels) = 0;
        virtual Tensor* backward(Tensor*& gradients) = 0;
};

#endif /* !BASELOSS_CUH */