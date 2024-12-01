#pragma once
#ifndef MODULE_H
#define MODULE_H

#include <string>
#include "tensor.cuh"

class Module {
    public:
        bool is_training;
        std::string prefix;
    public:
        Module* train();
        Module* eval();
};


#endif /* !MODULE_H */