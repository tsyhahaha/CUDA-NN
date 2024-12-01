#pragma once
#ifndef BASEMODEL_H
#define BASEMODEL_H

#include "tensor.cuh"
#include "layers.cuh"
#include "configure.cuh"

class BaseModel: public Module {
    public:
        Tensor* input;
        Tensor* mask;

    public:
        virtual Tensor* forward(Tensor* data, Tensor* mask) = 0;
        virtual Tensor* backward(Tensor* gradients) = 0;
        virtual void name_params(std::map<std::string, Tensor*>& np) = 0;
};

#endif /* !BASEMODEL_H */