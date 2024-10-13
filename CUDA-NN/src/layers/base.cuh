#pragma once
#ifndef BASE_H
#define BASE_H

#include "tensor.cuh"

class BaseLayer {
    protected:
        std::string prefix;
        Tensor* weights = nullptr;
        Tensor* bias = nullptr;

        Tensor* input;
        Tensor* output;
        Tensor* outputBackward;
        // Tensor* deltaWeights;
        // Tensor* deltaBias;
    public:
        Tensor* getWeights();
        Tensor* getBias();
        // Tensor* getDeltaWeights();
        // Tensor* getDeltaBias();

        // utils
        void load_weights(float *h_data, DimVector shape, const std::string& target);
        
        void load_weights(float *h_weight_data, float *h_bias_data, DimVector weight_shape, DimVector bias_shape);

        void load_weights();

        virtual Tensor* forward(Tensor* data) = 0;
        // virtual Tensor* backward(Tensor* gradients) = 0;
};

#endif /* !BASE_H */