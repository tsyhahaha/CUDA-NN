#ifndef OPT_CUH
#define OPT_CUH

#include "models.cuh"
#include "tensor.cuh"
#include "configure.cuh"

enum ScheduleType {
    LINEAR
};

class Optimizer {
    public:
        std::map<std::string, Tensor*> name_params;
        float lr;
        int steps = 0;
        int max_steps; // 最大步数
        ScheduleType type;

    public:
        float get_lr() {
            if (type == LINEAR) {
                // lr = lr_init * (1 - steps / max_steps)
                float decay = static_cast<float>(steps) / max_steps;
                return lr * (1 - decay);
            }
            return lr;
        }

        virtual void step() = 0;
        virtual void zero_grad() = 0;
};

#endif /* !OPT_CUH */