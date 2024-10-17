#pragma once
#ifndef BASE_H
#define BASE_H

#include "tensor.cuh"
#include "module.cuh"

class BaseLayer: public Module {
    protected:
        Tensor* weights = nullptr;
        Tensor* bias = nullptr;

        Tensor* input = nullptr;
        Tensor* output = nullptr;
        Tensor* outputBackward = nullptr;
        // Tensor* deltaWeights;
        // Tensor* deltaBias;
    public:
        Tensor* getWeights();
        Tensor* getBias();
        // Tensor* getDeltaWeights();
        // Tensor* getDeltaBias();

        // utils
        void load_weights(float *h_data, size_t n_data, const std::string& target);
        
        void load_weights(float *h_weight_data, float *h_bias_data, size_t n_data_weights, size_t n_data_bias);

        void load_weights(std::vector<float>& params, const std::string& target);

        void load_weights();
        void reset();
};

#endif /* !BASE_H */