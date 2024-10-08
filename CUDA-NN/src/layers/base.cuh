#pragma once
#ifndef BASE_H
#define BASE_H

#include "tensor.cuh"

class BaseLayer {
    protected:
        Tensor* weights = nullptr;
        Tensor* bias = nullptr;
        // Tensor* deltaWeights;
        // Tensor* deltaBias;
    public:
        Tensor* getWeights();
        Tensor* getBias();
        // Tensor* getDeltaWeights();
        // Tensor* getDeltaBias();

        virtual Tensor* forward(Tensor* data) = 0;
        // virtual Tensor* backward(Tensor* gradients) = 0;
};

#endif