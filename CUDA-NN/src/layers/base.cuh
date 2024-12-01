#pragma once
#ifndef BASELAYER_H
#define BASELAYER_H

#include "tensor.cuh"
#include "module.cuh"
#include "configure.cuh"

class BaseLayer: public Module {
    protected:
        Tensor* weights = nullptr;
        Tensor* bias = nullptr;

        Tensor* input = nullptr;
        Tensor* output = nullptr;

        Tensor* d_out = nullptr;
        Tensor* d_in = nullptr;
        Tensor* d_weights = nullptr;
        Tensor* d_bias = nullptr;
    public:
        ~BaseLayer();
        Tensor* getWeights();
        Tensor* getBias();

        Tensor* getDeltaWeights();
        Tensor* getDeltaBias();

        // utils
        void load_weights(float *h_data, size_t n_data, const std::string& target);
        void load_weights(float *h_weight_data, float *h_bias_data, size_t n_data_weights, size_t n_data_bias);
        void load_weights(std::vector<float>& params, const std::string& target);

        void load_weights();
        void init_weights();
        void name_params(std::map<std::string, Tensor*>& np);
        void reset();

        void train();
        void eval();

        virtual Tensor* forward(Tensor* data) = 0;
        virtual Tensor* backward(Tensor* gradients) = 0;
};

#endif /* !BASELAYER_H */