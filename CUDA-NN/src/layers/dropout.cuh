#pragma once
#ifndef DROPOUT_H
#define DROPOUT_H

#include "tensor.cuh"
#include "base.cuh"

/* 
https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html 
torch.nn.ReLU(inplace=False)
*/
class Dropout: public BaseLayer {
    private:
        float p;
        bool inplace;   // If true, input from this layer cannot be reused.
        
        Tensor* input=nullptr;
        Tensor* output=nullptr;
        Tensor* outputBackward=nullptr;

    public:
        Dropout(float p = 0.5f, bool inplace = false);

        Tensor* forward(Tensor* data);
        Tensor* backward(Tensor* gradients);
};

#endif