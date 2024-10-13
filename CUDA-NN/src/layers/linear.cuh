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
        int in_features;
        int out_features; 

        // Tensor* weights; // (out_features, in_features)
        // Tensor* bias;   // (out_features)
        
        // Tensor* input=nullptr;      // (*, in_features)
        // Tensor* output=nullptr;     // (*, out_features)
        // Tensor* outputBackward=nullptr;

    public:
        Linear(std::string prefix, size_t in_features, size_t out_features, bool bias=true, InitType init_type=KAIMING);
        ~Linear();

        Tensor* forward(Tensor* data);
        Tensor* backward(Tensor* gradients);

};

#endif /* !LINEAR_H */