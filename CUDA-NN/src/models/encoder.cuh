#ifndef ENCODER_H
#define ENCODER_H

#include "base.cuh"
#include "kernels.cuh"
#include "tensor.cuh"
#include "layers.cuh"
#include "stn3d.cuh"
#include "stnkd.cuh"

class Encoder: public BaseModel {
    private:
        bool global_feat;
        bool feature_transform;
        size_t channel;
        Tensor* feat;
        Tensor* trans_points = nullptr;
        Tensor* f_trans = nullptr;
        Tensor* p_trans = nullptr;
        Tensor* trans_feat = nullptr;
        Tensor* output = nullptr;

        Tensor* max_index = nullptr;
        Tensor* max_gradients = nullptr;

        Tensor* f_gradients = nullptr;
        Tensor* p_gradients = nullptr;
        Tensor* trans_feat_gradients = nullptr;
        Tensor* trans_points_gradients = nullptr;

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
        void init_weights();
        Encoder* train();
        Encoder* eval();

        Tensor* forward(Tensor* data, Tensor* mask);
        Tensor* backward(Tensor* gradients);
        void name_params(std::map<std::string, Tensor*>& np);
};

#endif /* !ENCODER_H */