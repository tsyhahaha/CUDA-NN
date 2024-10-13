#include "base.cuh"

Tensor* BaseLayer::getWeights() {
    return this->weights;
}

Tensor* BaseLayer::getBias() {
    return this->bias;
}

void BaseLayer::load_weights() {
    this->weights->fromVec(Configurer::getWeights(this->prefix + "weight"));
    this->bias->fromVec(Configurer::getWeights(this->prefix + "bias"));
}

// Tensor* Layer::getDeltaweightss() {
//     return this->deltaweightss;
// }

// Tensor* Layer::getDeltaBias() {
//     return this->deltaBias;
// }