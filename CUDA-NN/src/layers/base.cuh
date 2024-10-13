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

        // utils
        void load_weights(float *h_data, DimVector shape, const std::string& target);
        
        void load_weights(float *h_weight_data, float *h_bias_data, DimVector weight_shape, DimVector bias_shape);

        virtual Tensor* forward(Tensor* data) = 0;
        // virtual Tensor* backward(Tensor* gradients) = 0;
};

#endif /* !BASE_H */