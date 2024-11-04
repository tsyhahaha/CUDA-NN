#ifndef ENCODER_H
#define ENCODER_H

#include "base.cuh"
#include "tensor.cuh"
#include "layers.cuh"
#include "stn3d.cuh"
#include "stnkd.cuh"

class Encoder: public BaseModel {
    private:
        bool global_feat;
        bool feature_transform;
        size_t channel;
        Tensor* p_trans = nullptr;
        Tensor* f_trans = nullptr;
        Tensor* output = nullptr;

    public:
        STN3d* stn;
        STNkd* fstn;
        Conv1d* conv1;
        Conv1d* conv2;
        Conv1d* conv3;
        BatchNorm1d* bn1;
        BatchNorm1d* bn2;
        BatchNorm1d* bn3;
        ReLU* relu;

    public:
        Encoder(std::string prefix, bool global_feat = true, bool feature_transform = false, size_t channel = 3);
        ~Encoder();

        void load_weights();

        Tensor* forward(Tensor* data, Tensor* mask);
        Tensor* backward(Tensor* gradients);
};

#endif /* !ENCODER_H */