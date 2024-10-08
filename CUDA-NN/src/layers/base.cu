#include "base.cuh"

Tensor* BaseLayer::getWeights() {
    return this->weights;
}

Tensor* BaseLayer::getBias() {
    return this->bias;
}

// Tensor* Layer::getDeltaWeights() {
//     return this->deltaWeights;
// }

// Tensor* Layer::getDeltaBias() {
//     return this->deltaBias;
// }