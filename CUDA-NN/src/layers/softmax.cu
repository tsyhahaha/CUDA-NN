#include "softmax.cuh"

SoftMax::SoftMax(size_t dim) {
    this->dim = dim;
}

Tensor* SoftMax::forward(Tensor* data) {
    // this->input = data;
    // Tensor* x_max = data->max(this->dim);
    // Tensor* z = data->sub(x_max);
    // Tensor* nominator = z->exp();
    // Tensor* denominator = nominator->sum(this->dim);
    // this->output = nominator->div(denominator);
    // delete x_max, z, nominator, denominator;
    // return this->output;

    return data;
}

Tensor* SoftMax::backward(Tensor* gradients) {
    return nullptr;
} 