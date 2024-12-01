#include "sgd.cuh"

SGD::SGD(std::map<std::string, Tensor*>& name_params, float lr, float momentum) {
    this->name_params = name_params;
    this->momentum = momentum;
    this->lr = lr;

    if(momentum < 1.0f && momentum > 0.0f) {
        for(auto& pair: name_params) {
            grads_cache[pair.first] = new Tensor(pair.second->getShape(), ZERO);
        }
    }
}
SGD::~SGD() {
    for (auto& pair : name_params) {
        delete pair.second;
        pair.second = nullptr;
    }

    for (auto& pair : grads_cache) {
        delete pair.second;
        pair.second = nullptr;
    }
}

void SGD::step() {
    bool exist_nan = false;
    for(auto& pair: name_params) {
        std::string name = pair.first;
        // DEBUG_PRINT("[SGD] update: %s\n", name.c_str());
        Tensor* grads_acc = name_params[name]->getGradsAcc();
        if(Configurer::hook) {
            if(!grads_acc) {
                ERROR("%s gradients is null!\n", name.c_str());
            }
            if(grads_acc->checkNan()) {
                DEBUG_PRINT("%s gradients exist Nan!\n", name.c_str());
                exist_nan = true;
            }
        }
        Tensor* trainable_tensor = name_params[name];
        grads_cache[name]->add_(grads_acc, momentum, 1.0f);
        trainable_tensor->add_(grads_cache[name], 1.0f, get_lr());
    }
    if(exist_nan) {
        ERROR("Please check the nan grads!\n");
    }
    this->steps += 1;
}

void SGD::zero_grad() {
    DEBUG_PRINT("[SGD] zero_grad()\n");
    for(auto& pair: name_params) {
        std::string name = pair.first;
        name_params[name]->getGradsAcc()->initialize(0.0f);
        grads_cache[name]->initialize(0.0f);
    }
}
