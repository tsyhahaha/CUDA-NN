#pragma once
#ifndef DROPOUT_H
#define DROPOUT_H

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "base.cuh"

class Dropout: public BaseLayer {
    private:
        float p;
        Tensor* mask;
        bool inplace;   // If true, input from this layer cannot be reused.
        
        // Tensor* input=nullptr;
        // Tensor* output=nullptr;
        // Tensor* outputBackward=nullptr;

    public:
        Dropout(std::string prefix="", float p = 0.5f, bool inplace = false);
        Dropout(float p = 0.5f, bool inplace = false);

        Tensor* forward(Tensor* data);
        Tensor* backward(Tensor* gradients);
};

#endif /* !DROPOUT_H */