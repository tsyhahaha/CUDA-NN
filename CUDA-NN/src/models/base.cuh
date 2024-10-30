#pragma once
#ifndef BASEMODEL_H
#define BASEMODEL_H

#include "tensor.cuh"
#include "layers.cuh"

class BaseModel: public Module {
    protected:

    public:
        virtual Tensor* forward(Tensor* data, Tensor* mask) = 0;
        virtual Tensor* backward(Tensor* gradients) = 0;
};

#endif /* !BASEMODEL_H */