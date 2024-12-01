#ifndef ADAM_CUH
#define ADAM_CUH

#include "base_opt.cuh"

class Adam: public Optimizer {
    public:
        float beta1;
        float beta2;
        float eps;
        
    public:
        Adam(std::map<std::string, Tensor*>& name_params, float lr=1e-3, float beta1=0.99, float beta2=0.999, float eps=1e-8);
        ~Adam();
};

#endif /* !ADAM_CUH */