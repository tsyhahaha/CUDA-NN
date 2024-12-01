#ifndef CE_CUH
#define CE_CUH

#include "base_loss.cuh"

class CrossEntropyLoss: public BaseLoss {
    public:
        std::string reduction;
    public:
        CrossEntropyLoss(std::string reduction = "mean");
        ~CrossEntropyLoss();

        float forward(Tensor* logits, Tensor* labels) override;
        Tensor* backward(Tensor*& gradients) override;
};

#endif /* !CE_CUH */