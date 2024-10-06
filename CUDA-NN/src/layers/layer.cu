#include "layer.cuh"

Tensor* Layer::getWeights() {
    return this->weights;
}

Tensor* Layer::getBias() {
    return this->bias;
}

// Tensor* Layer::getDeltaWeights() {
//     return this->deltaWeights;
// }

// Tensor* Layer::getDeltaBias() {
//     return this->deltaBias;
// }