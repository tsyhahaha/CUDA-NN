#ifndef STN3D_H
#define STN3D_H

#include "configure.cuh"
#include "tensor.cuh"
#include "layers.cuh"
#include "base.cuh"

/* Maybe it can be acquired by STNkd...... */
class STN3d: public BaseModel {
    private:
        Tensor* iden;
        Tensor* o = nullptr;
        Tensor* output=nullptr;

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
        STN3d(std::string prefix, size_t channel);
        ~STN3d();

        void load_weights();

        Tensor* forward(Tensor* data, Tensor* mask);
        Tensor* backward(Tensor* gradients);
};

#endif /* !STN3D_H */