#include "adam.cuh"

Adam::Adam(std::map<std::string, Tensor*>& name_params, float lr, float beta1, float beta2, float eps) {
    this->name_params = name_params;
    this->lr = lr;
    
    this->beta1 = beta1;
    this->beta2 = beta2;
    this->eps = eps;
}

Adam::~Adam(){}