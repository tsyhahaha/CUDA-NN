#ifndef STNKD_H
#define STNKD_H

#include "tensor.cuh"
#include "layers.cuh"
#include "base.cuh"

class STNkd: public BaseModel {
    private:
        size_t k;
        
    public:
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
        STNkd(std::string prefix, size_t k=64);
        ~STNkd();

        void load_weights();

        Tensor* forward(Tensor* data, Tensor* mask=nullptr);
        Tensor* backward(Tensor* gradients);
};

#endif /* !STNKD_H */