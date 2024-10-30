#include "dropout.cuh"
#include "kernels.cuh"

__global__
void kDropout1D(float* d_in, float* d_out, int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < N) {
        if(d_in[col] < 0.0f) {
            d_out[col] = 0.0f;
        } else {
            d_out[col] = d_in[col];
        }
    }
}

Dropout::Dropout(std::string prefix, float p, bool inplace) {
    this->p = p;
    this->inplace = inplace;
    this->input = nullptr;

    // Prepare output for forward and backprop
    this->output = nullptr;
    this->outputBackward = nullptr;

    this->prefix = prefix;
}

Dropout::Dropout(float p, bool inplace) {
    this->p = p;
    this->inplace = inplace;
    this->input = nullptr;

    // Prepare output for forward and backprop
    this->output = nullptr;
    this->outputBackward = nullptr;
}


Tensor* Dropout::forward(Tensor* data) {
    this->reset();
    if(this->is_training)
        this->input = data;

    DimVector shape_o = data->getShape(); 
    int dim = shape_o.size();
    size_t n_data = data->getDataNum();

    int block = BLOCK_SIZE1D;
    int grid = (n_data - 1) / block + 1;

    if (inplace) {
        kDropout1D<<<grid, block>>>(data->getData(), data->getData(), n_data);
        this->output = data;
    } else {
        Tensor* tensor_o = new Tensor(shape_o);
        kDropout1D<<<grid, block>>>(data->getData(), tensor_o->getData(), n_data);
        this->output = tensor_o;
    }

    return this->output;
}
 
Tensor* Dropout::backward(Tensor* gradients) {
    
    return nullptr;
}