#pragma once
#ifndef MODULE_H
#define MODULE_H

#include <string>
#include "tensor.cuh"

class Module {
    protected:
        bool is_training = false;
        std::string prefix;
    public:

        void train();
        void eval();
};


#endif /* !MODULE_H */