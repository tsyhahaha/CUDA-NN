#include "module.cuh"

Module* Module::train() {
    this->is_training = true;
    return this;
}

Module* Module::eval() {
    this->is_training = false;
    return this;
}