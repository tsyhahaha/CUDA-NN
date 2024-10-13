#ifndef POINTNET_H
#define POINTNET_H

#include "module.cuh"
#include "tensor.cuh"
#include "layers.cuh"

#include "encoder.cuh"

class PointNet: public Module {
    private:
        Encoder* feat;
        Linear* fc1;
        Linear* fc2;
        Linear* fc3;
        Dropout* dropout;
        BatchNorm1d* bn1;
        BatchNorm1d* bn2;
        SoftMax* softmax;
        ReLU* relu;

    public:
        PointNet(size_t k=10, bool normal_channel=false);
        ~PointNet();

        void load_weights();

        Tensor* forward(Tensor* data);
        Tensor* backward(Tensor* gradients);
};

#endif /* !POINTNET_H */