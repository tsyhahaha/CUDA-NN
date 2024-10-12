#ifndef SFM_H
#define SFM_H

#include "tensor.cuh"
#include "base.cuh"

class SoftMax: public BaseLayer {
    private:
        size_t dim;

        Tensor* input;
        Tensor* output;
        Tensor* outputBackward;

    public:
        SoftMax(size_t dim);
        ~SoftMax();

        Tensor* forward(Tensor* data);
        Tensor* backward(Tensor* gradients); 

};

#endif