#include "softmax.cuh"

SoftMax::SoftMax(size_t dim) {
    this->dim = dim;
}

Tensor* SoftMax::forward(Tensor* data) {
    this->input = data;

    return data;
}

Tensor* SoftMax::backward(Tensor* gradients) {
    return nullptr;
} 