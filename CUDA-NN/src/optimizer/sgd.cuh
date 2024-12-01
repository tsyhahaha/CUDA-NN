#ifndef SGD_CUH
#define SGD_CUH

#include "base_opt.cuh"

class SGD: public Optimizer{
    private:
        std::map<std::string, Tensor*> grads_cache;
        float momentum;

    public:
        SGD(std::map<std::string, Tensor*>& name_params, float lr, float momentum);
        ~SGD();

        void step();
        void zero_grad();
};

#endif /* !SGD_CUH */