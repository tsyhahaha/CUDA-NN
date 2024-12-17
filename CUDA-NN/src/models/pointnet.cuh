#ifndef POINTNET_H
#define POINTNET_H

#include "utils.cuh"
#include "configure.cuh"
#include "tensor.cuh"
#include "layers.cuh"
#include "encoder.cuh"

class PointNet: public BaseModel {
    public:
        Encoder* feat;
        Linear* fc1;
        Linear* fc2;
        Linear* fc3;
        Dropout* dropout;
        BatchNorm1d* bn1;
        BatchNorm1d* bn2;
        SoftMax* softmax;
        ReLU* relu;

    private:
        Tensor* d_out;

    public:
        PointNet(std::string prefix, size_t k=10, bool normal_channel=false);
        PointNet(size_t k=10, bool normal_channel=false);
        ~PointNet();

        void load_weights();
        void init_weights();
        PointNet* train();
        PointNet* eval();

        Tensor* forward(Tensor* data, Tensor* mask);
        Tensor* backward(Tensor* gradients);
        void name_params(std::map<std::string, Tensor*>& np);
};

#endif /* !POINTNET_H */