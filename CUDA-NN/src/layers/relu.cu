#include "relu.cuh"
#include "kernels.cuh"

__global__
void kReLU1D(float* A, float* d_out, int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < N) {
        d_out[col] = fmaxf(0.0f, A[col]);
    }
}

__global__
void kReLU2D(float* A, float* d_out, int M, int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < M && col < N) {
        if (A[row * N + col] < 0.0f) {
            d_out[row * N + col] = 0.0f;
        } else {
            d_out[row * N + col] = A[row * N + col];
        }
    }
}


ReLU::ReLU(bool inplace) {
    this->inplace = inplace;
    this->input = nullptr;
}

ReLU::ReLU(std::string prefix, bool inplace) {
    this->inplace = inplace;
    this->input = nullptr;

    this->prefix = prefix;
}

Tensor* ReLU::forward(Tensor* data) {
    DimVector shape_o = data->getShape();
    if(this->output == nullptr) {
        this->output = new Tensor(shape_o);
    }
    this->output->reset(shape_o);

    if(this->is_training)
        this->input = data;

    int dim = shape_o.size();
    size_t n_data = data->getSize();

    int block = BLOCK_SIZE1D;
    int grid = (n_data - 1) / block + 1;

    if (inplace) {
        kReLU1D<<<grid, block>>>(data->getData(), data->getData(), n_data); CHECK_KERNEL();
        this->output = data;
    } else {
        kReLU1D<<<grid, block>>>(data->getData(), this->output->getData(), n_data);CHECK_KERNEL();
    }

    return this->output;
}
 
Tensor* ReLU::backward(Tensor* gradients) {
    return nullptr;
}