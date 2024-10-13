#ifndef CONV1D_H
#define CONV1D_H

#include "base.cuh"
#include "tensor.cuh"

/*
https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)

However, the implementations simplifies for PointNet.
*/
class Conv1d: public BaseLayer {
    private:
        size_t in_channels;         // the channel of kernels
        size_t out_channels;        // num kernels
        size_t kernel_size;         // ks == 1 in PointNet

        Tensor* weights;         // (C_out x C_in) each col -> each kernel
        Tensor* bias;            // (C_out)

        Tensor* input=nullptr;      // (N, C_in, L_in) or (C_in, L_in)
        Tensor* output=nullptr;     // (N, C_out, L_in) or (C_out, L_in) if ks = 1
        Tensor* outputBackward=nullptr;

    public:
        Conv1d(size_t in_channels, size_t out_channels, size_t kernel_size, bool bias = true);
        ~Conv1d();

        void load_weights(float *h_weights_data, float *h_bias_data, DimVector weights_shape, DimVector bias_shape);

        Tensor* forward(Tensor* data);
        Tensor* backward(Tensor* gradients);
};

 

#endif /* !CONV1D_H */