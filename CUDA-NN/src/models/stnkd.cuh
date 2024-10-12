#ifndef STNKD_H
#define STNKD_H

#include "tensor.cuh"
#include "layers.cuh"
#include "module.cuh"

class STNkd: public Module {
    private:
        size_t k;

        Conv1d* conv1;
        Conv1d* conv2;
        Conv1d* conv3;
        Linear* fc1;
        Linear* fc2;
        Linear* fc3;
        ReLU* relu;
        BatchNorm1d* bn1;
        BatchNorm1d* bn2;
        BatchNorm1d* bn3;
        BatchNorm1d* bn4;
        BatchNorm1d* bn5;

    public:
        STNkd(size_t k);
        ~STNkd();

        void load_weights();

        Tensor* forward(Tensor* data);
        Tensor* backward(Tensor* gradients);
};

#endif