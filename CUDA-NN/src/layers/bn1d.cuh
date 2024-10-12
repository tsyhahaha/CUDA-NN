#ifndef BN1D_H
#define BN1D_H

#include "base.cuh"
#include "tensor.cuh"

/*
https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html
torch.nn.BatchNorm1d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None)

Applies Batch Normalization over a 2D or 3D input.
*/
class BatchNorm1d: public BaseLayer {
    private:
        size_t num_features;        // the num of features/channels, C
        float eps;
        float momentum; // x^_new = (1-momentum)*x^ + monmentum*x_t (ob)
        bool affine;    // alpha, beta
        bool track_running_stats;
        
        
        Tensor* running_mean;   // the mean per channel(C)
        Tensor* running_var;    // the var  per channel(C)
        Tensor* weights;     // affine gamma (C)
        Tensor* bias;        // affine beta  (C)

        Tensor* input=nullptr;      // (N, C) or (N, C, L)
        Tensor* output=nullptr;     // (N, C) or (N, C, L)
        Tensor* outputBackward=nullptr;

    public:
        BatchNorm1d(size_t num_features, float eps = 1e-5, float monmentum=0.1, bool affine=true, bool track_running_stats=true);
        ~BatchNorm1d();

        void load_weights(float *h_weights_data, float *h_bias_data, DimVector weights_shape, DimVector bias_shape);

        void load_weights(float *h_data, DimVector shape, const std::string& target);

        Tensor* forward(Tensor* data);
        Tensor* backward(Tensor* gradients);

    private:
        
};



#endif