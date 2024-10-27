#pragma once
#ifndef SFM_H
#define SFM_H

#include "tensor.cuh"
#include "base.cuh"

class SoftMax: public BaseLayer {
    private:
        size_t dim;
        bool apply_log;

    public:
        SoftMax(std::string prefix, size_t dim, bool apply_log=false);
        SoftMax(size_t dim, bool apply_log=false);
        ~SoftMax();

        Tensor* forward(Tensor* data);
        Tensor* backward(Tensor* gradients); 

};

#endif /* !SFM_H */