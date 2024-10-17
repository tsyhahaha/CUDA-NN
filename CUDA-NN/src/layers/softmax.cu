#include "softmax.cuh"

SoftMax::SoftMax(std::string prefix, size_t dim) {
    this->dim = dim;
    this->prefix = prefix;
}

SoftMax::SoftMax(size_t dim) {
    this->dim = dim;
}

SoftMax::~SoftMax() {
    delete this->input;
    delete this->output;
}

Tensor* SoftMax::forward(Tensor* data) {
    this->reset();
    if(this->is_training)
        this->input = data;
    
    Tensor* x_max = data->max(this->dim);
    Tensor* z = data->sub(x_max);
    Tensor* nominator = z->exp();
    Tensor* denominator = nominator->sum(this->dim);
    this->output = nominator->div(denominator);
    this->output->log_();
    delete x_max, z, nominator, denominator;
    return this->output;
}

Tensor* SoftMax::backward(Tensor* gradients) {
    return nullptr;
} 