#include "base.cuh"

BaseLayer::~BaseLayer(){
    if(weights != nullptr) {
        delete weights;
    }
    if(bias != nullptr) {
        delete bias;
    }
    if(input != nullptr) {
        delete input;
    }
    if(output != nullptr) {
        delete output;
    }
    if(d_out != nullptr) {
        delete d_out;
    }
    if(d_weights != nullptr) delete d_weights;
    if(d_bias != nullptr) delete d_bias;
}

Tensor* BaseLayer::getWeights() {
    return this->weights;
}

Tensor* BaseLayer::getBias() {
    return this->bias;
}

void BaseLayer::load_weights() {
    this->weights->fromVec(Configurer::getWeights(this->prefix + "weight"));
    if(bias) {
        this->bias->fromVec(Configurer::getWeights(this->prefix + "bias"));
    }
}

void BaseLayer::init_weights() {
    this->weights->initialize(is_training ? RANDOM:NONE);
    if(bias) {
        this->bias->initialize(is_training ? RANDOM:NONE);
    }
}

void BaseLayer::load_weights(float *h_data, size_t n_data, const std::string& target) {
    if(target=="weights") {
        assert(n_data == this->weights->getSize());
        this->weights->load(h_data, n_data);
    } else if(target == "bias") {
        assert(n_data == this->bias->getSize());
        this->bias->load(h_data, n_data);
    } else {
        ERROR("Load weights %s error!\n", target.c_str());
    }
}

void BaseLayer::load_weights(float *h_weight_data, float *h_bias_data, size_t n_data_weights, size_t n_data_bias) {
    assert(n_data_weights == this->weights->getSize());
    assert(n_data_bias == this->bias->getSize());
    this->weights->load(h_weight_data, n_data_weights);
    this->bias->load(h_bias_data, n_data_bias);
}

void BaseLayer::load_weights(std::vector<float>& params, const std::string& target) {
    size_t n_data = params.size();
    float* h_data = params.data();
    if(target=="weights") {
        if(n_data != this->weights->getSize()) {
            ERROR("weights data num not match: %ld != %ld!\n", n_data, this->weights->getSize());
        }
        this->weights->fromVec(params);
    } else if(target == "bias") {
        if(n_data != this->bias->getSize()) {
            ERROR("bias data num not match: %ld != %ld!\n", n_data, this->bias->getSize());
        }
        this->bias->fromVec(params);
    } else {
        ERROR("Load weights %s error!\n", target.c_str());
    }
}

void BaseLayer::name_params(std::map<std::string, Tensor*>& np) {
    if(this->weights)
        np.insert(std::make_pair(prefix + "weights", weights));
    if(this->bias)
        np.insert(std::make_pair(prefix + "bias", bias));
}

void BaseLayer::reset(){
    if(this->output != nullptr) {
        delete this->output;
        DEBUG_PRINT("FREE OUTPUT\n");
        this->output = nullptr;
    }
}

void BaseLayer::train() {
    this->is_training = true;
    if(weights)
        this->weights->train();
    if(bias)
        this->bias->train();
}

void BaseLayer::eval(){
    this->is_training = false;
    if(weights)
        this->weights->eval();
    if(bias)
        this->bias->eval();
}