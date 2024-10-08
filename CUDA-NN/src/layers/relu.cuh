#pragma once
#ifndef RELU_H
#define RELU_H

#include "tensor.cuh"
#include "base.cuh"

class ReLU: public BaseLayer {
    private:
        bool inplace;   // If true, input from this layer cannot be reused.
        
        Tensor* input;
        Tensor* output;
        Tensor* outputBackward;

    public:
        ReLU(bool inplace = false);

        Tensor* forward(Tensor* data);
        Tensor* backward(Tensor* gradients);
};

#endif