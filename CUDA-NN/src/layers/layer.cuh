#pragma once
#ifndef LAYER_H
#define LAYER_H

#include "../tensor/tensor.cuh"

class Layer {
    protected:
        Tensor* weights;
        Tensor* bias;
        Tensor* deltaWeights;
        Tensor* deltaBias;
    public:
        Tensor* getWeights();
        Tensor* getBias();
        // Tensor* getDeltaWeights();
        // Tensor* getDeltaBias();

        virtual Tensor* forward(Tensor* data) = 0;
        // virtual Tensor* backward(Tensor* gradients) = 0;
};

#endif  /* !LAYER_HPP */