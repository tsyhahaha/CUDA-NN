#ifndef NLL_CUH
#define NLL_CUH

#include "base_loss.cuh"

class NegativeLikelihoodLoss: public BaseLoss {
    public:
        std::string reduction;
        cudaStream_t s;

        Tensor* logits = nullptr;
        Tensor* labels = nullptr;

    public:
        NegativeLikelihoodLoss(std::string reduction = "mean");

        ~NegativeLikelihoodLoss();

        float forward(Tensor* logits, Tensor* labels) override;
        Tensor* backward(Tensor*& gradients) override;
};

#endif /* !NLL_CUH */