#pragma once
#ifndef LINEAR_H
#define LINEAR_H

#include <string>
#include <cstring> // For strcmp

#include "../tensor/tensor.cuh"
#include "../common/utils.cuh"
#include "layer.cuh"


class Linear: public Layer {
    private:
        int in_features;
        int out_features; 

        Tensor* weight;
        Tensor* bias;
        
        Tensor* input;    // saved for backward
        Tensor* output;
        Tensor* outputBackward;

    public:
        Linear(size_t in_features, size_t out_features, bool bias, InitType init_type);

        void load_weights(float *h_data, DimVector shape, const std::string& target);
        void load_weights(float *h_weight_data, float *h_bias_data, DimVector weight_shape, DimVector bias_shape);

        Tensor* forward(Tensor* data);
        Tensor* backward(Tensor* gradients);

};

#endif