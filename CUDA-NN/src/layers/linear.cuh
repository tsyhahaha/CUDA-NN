#pragma once
#ifndef LINEAR_H
#define LINEAR_H

#include <string>
#include <cstring> // For strcmp

#include "tensor.cuh"
#include "base.cuh"


/* 
https://pytorch.org/docs/stable/generated/torch.nn.Linear.html 
torch.nn.Linear(in_features, out_features, bias=True, device=None, dtype=None)
*/
class Linear: public BaseLayer {
    private:
        size_t in_features;
        size_t out_features; 

    public:
        Linear(std::string prefix, size_t in_features, size_t out_features, bool bias=true);
        Linear(size_t in_features, size_t out_features, bool bias=true);

        void init_weights();
        Linear* train();
        Tensor* forward(Tensor* data);
        Tensor* backward(Tensor* gradients);
};

#endif /* !LINEAR_H */