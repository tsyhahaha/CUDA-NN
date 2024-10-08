#ifndef CONV1D_H
#define CONV1D_H

#include "base.cuh"
#include "tensor.cuh"

/*
torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
*/
class Conv1d: public BaseLayer {
    private:
        size_t in_channels;        // the channel of kernels
        size_t out_channels;       // num kernels
        size_t kernel_size;

        Tensor* weight;         // (out_channels x in_channels) each col -> each kernel
        Tensor* bias = nullptr;

        Tensor* input;    // saved for backward
        Tensor* output;
        Tensor* outputBackward;

    public:
        Conv1d(size_t in_channels, size_t out_channels, size_t kernel_size, bool bias = true);

        void load_weights(float* h_data, DimVector& shape, const std::string& target);
        void load_weights(float *h_weight_data, float *h_bias_data, DimVector weight_shape, DimVector bias_shape);

        Tensor* forward(Tensor* data);
        Tensor* backward(Tensor* gradients);
};



#endif