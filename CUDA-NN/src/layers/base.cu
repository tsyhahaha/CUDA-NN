#include "base.cuh"

Tensor* BaseLayer::getWeights() {
    return this->weights;
}

Tensor* BaseLayer::getBias() {
    return this->bias;
}

void BaseLayer::load_weights(float *h_weights_data, float *h_bias_data, DimVector weights_shape, DimVector bias_shape) {
    this->weights->initialize(h_weights_data, weights_shape);        
    this->bias->initialize(h_bias_data, bias_shape);
}

void BaseLayer::load_weights(float *h_data, DimVector shape, const std::string& target) {
    if(target == "weights") {
        assert(this->weights->getShape() == shape);
        this->weights->initialize(h_data, shape);        
    } else if(target == "bias") {
        assert(this->bias->getShape() == shape);
        this->bias->initialize(h_data, shape);
    }
}

// Tensor* Layer::getDeltaweightss() {
//     return this->deltaweightss;
// }

// Tensor* Layer::getDeltaBias() {
//     return this->deltaBias;
// }