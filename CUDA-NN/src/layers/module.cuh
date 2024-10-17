#pragma once
#ifndef MODULE_H
#define MODULE_H

#include <string>
#include "tensor.cuh"

class Module {
    protected:
        bool is_training;
        std::string prefix;
    public:

        void train();
        void eval();

        virtual Tensor* forward(Tensor* data) = 0;
        virtual Tensor* backward(Tensor* gradients) = 0;

};


#endif /* !MODULE_H */