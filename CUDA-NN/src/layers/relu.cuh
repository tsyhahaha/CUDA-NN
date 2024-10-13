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
        
        Tensor* input=nullptr;
        Tensor* output=nullptr;
        Tensor* outputBackward=nullptr;

    public:
        ReLU(bool inplace = false);
        ~ReLU();

        Tensor* forward(Tensor* data);
        Tensor* backward(Tensor* gradients);
};

#endif /* !RELU_H */