#include "dropout.cuh"

__global__ void kDropout1D(float* input, float* output, float* mask, int n, float dropout_prob, unsigned long long seed) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        curandState state;
        curand_init(seed, index, 0, &state);

        float rand_val = curand_uniform(&state);

        if (rand_val < dropout_prob) {
            mask[index] = 0.0f;
            output[index] = 0.0f;
        } else {
            mask[index] = 1.0f;
            output[index] = input[index];
        }
    }
}

Dropout::Dropout(std::string prefix, float p, bool inplace) {
    this->p = p;
    this->inplace = inplace;

    this->prefix = prefix;
}

Dropout::Dropout(float p, bool inplace) {
    this->p = p;
    this->inplace = inplace;
}


Tensor* Dropout::forward(Tensor* data) {
    this->reset();
    if(this->is_training)
        this->input = data;

    unsigned long long seed = generateRandomSeed();

    DimVector shape_o = data->getShape(); 
    int dim = shape_o.size();
    size_t n_data = data->getSize();

    int block = BLOCK_SIZE1D;
    int grid = (n_data - 1) / block + 1;

    if (inplace) {
        kDropout1D<<<grid, block>>>(data->getData(), data->getData(), mask->getData(), n_data, p, seed);
        this->output = data;
    } else {
        Tensor* tensor_o = new Tensor(shape_o);
        kDropout1D<<<grid, block>>>(data->getData(), tensor_o->getData(), mask->getData(), n_data, p, seed);
        this->output = tensor_o;
    }

    return this->output;
}
 
Tensor* Dropout::backward(Tensor* gradients) {
    
    return nullptr;
}