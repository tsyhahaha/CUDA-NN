#include "module.cuh"

void Module::train() {
    this->is_training = true;
}

void Module::eval() {
    this->is_training = false;
}