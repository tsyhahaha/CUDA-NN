#include "relu.cuh"
#include "kernels.cuh"

__global__
void kReLU1D(float* A, float* d_out, int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < N) {
        if(A[col] < 0.0f) {
            d_out[col] = 0.0f;
        } else {
            d_out[col] = A[col];
        }
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


ReLU::ReLU(std::string prefix, bool inplace) {
    this->inplace = inplace;
    this->input = nullptr;

    this->prefix = prefix;

    // Prepare output for forward and backprop
    this->output = nullptr;
    this->outputBackward = nullptr;
}

ReLU::~ReLU() {
    delete input, output, outputBackward;
}

Tensor* ReLU::forward(Tensor* data) {
    this->input = data;

    DimVector shape_o = data->getShape(); // deep copy auto?
    int dim = shape_o.size();
    size_t n_data = data->getDataNum();

    int block = BLOCK_SIZE1D;
    int grid = (n_data - 1) / block + 1;

    if (inplace) {
        kReLU1D<<<grid, block>>>(data->getData(), data->getData(), n_data);
        this->output = data;
    } else {
        Tensor* tensor_o = new Tensor(shape_o);
        kReLU1D<<<grid, block>>>(data->getData(), tensor_o->getData(), n_data);
        this->output = tensor_o;
    }

    return this->output;
}
 
Tensor* ReLU::backward(Tensor* gradients) {
    
    return nullptr;
}