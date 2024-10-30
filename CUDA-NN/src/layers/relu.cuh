#pragma once
#ifndef RELU_H
#define RELU_H

#include "tensor.cuh"
#include "base.cuh"

/* 
https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html 
torch.nn.ReLU(inplace=False)
*/
class ReLU: public BaseLayer {
    private:
        bool inplace;   // If true, input from this layer cannot be reused.
        
    public:
        ReLU(std::string prefix, bool inplace = false);
        ReLU(bool inplace = false);

        Tensor* forward(Tensor* data);
        Tensor* backward(Tensor* gradients);
};

#endif /* !RELU_H */