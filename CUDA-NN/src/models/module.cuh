#ifndef MODULE_H
#define MODULE_H

#include <string>
#include "tensor.cuh"

class Module {
    protected:
        std::string prefix;
    public:

        virtual void load_weights() = 0;
        virtual Tensor* forward(Tensor* data) = 0;
        virtual Tensor* backward(Tensor* gradients) = 0;

};


#endif /* !MODULE_H */