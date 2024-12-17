#ifndef OPT_CUH
#define OPT_CUH

#include "models.cuh"
#include "tensor.cuh"
#include "configure.cuh"
#include "utils.cuh"

class Optimizer {
    public:
        std::map<std::string, Tensor*> name_params;
        float lr;
        int steps = 0;
        int step_size = 20;
        float gamma = 0.7;

    public:
        float get_lr() {
            // scheduler
            int step_count = steps / step_size;
            return lr * std::pow(gamma, step_count);
        }

        virtual void step() = 0;
        virtual void zero_grad() = 0;
};

#endif /* !OPT_CUH */