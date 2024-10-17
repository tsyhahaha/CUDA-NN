#include "bn1d.cuh"
#include "kernels.cuh"

/* [(N x C) - (C)] / sqrt(C + eps) * (C) + (C) */
__global__ 
void kBn1d_l2(float* d_data, float* d_out, float* weights, float* bias, 
    float* mean, float* var, float eps, int N, int C
) {
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;

    if(row >= N || col >= C) return;

    float scaling = rsqrtf(var[col] + eps); // 1 / sqrt(C + eps)
    float norm = (d_data[row * C + col] - mean[col]) * scaling;

    d_out[row * C + col] = norm * weights[col] + bias[col];
}


/* [(N x C x L) - (C)] / sqrt(C + eps) * (C) + (C) */
__global__ 
void kBn1d_l3(float* d_data, float* d_out, float* weights, float* bias, 
    float* mean, float* var, float eps, int N, int C, int L
) {
    int z = threadIdx.z + blockIdx.z * blockDim.z;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int x = threadIdx.x + blockIdx.x * blockDim.x;

    if(z >= N || y >= C || x >= L) return;

    float scaling = rsqrtf(var[y] + eps); // 1 / sqrt(C + eps)
    float norm = (d_data[z*C*L + y*L + x] - mean[y]) * scaling;

    d_out[z*C*L + y*L + x] = norm * weights[y] + bias[y];
}

void BatchNorm1d::load_weights() {
    this->weights->fromVec(Configurer::getWeights(this->prefix + "weight"));
    this->bias->fromVec(Configurer::getWeights(this->prefix + "bias"));
    this->running_mean->fromVec(Configurer::getWeights(this->prefix + "running_mean"));
    this->running_var->fromVec(Configurer::getWeights(this->prefix + "running_var"));
}

void BatchNorm1d::load_weights(std::vector<float>& params, const std::string& target) {
    size_t n_data = params.size();
    float* h_data = params.data();
    if(target=="weights") {
        assert(n_data == this->weights->getDataNum());
        this->weights->load(h_data, n_data);
    } else if(target == "bias") {
        assert(n_data == this->bias->getDataNum());
        this->bias->load(h_data, n_data);
    } else if(target == "mean") {
        assert(this->running_mean->getDataNum() == n_data);
        this->running_mean->load(h_data, n_data);
    } else if(target == "var") {
        assert(this->running_var->getDataNum() == n_data);
        this->running_var->load(h_data, n_data);
    } else {
        ERROR("Load weights %s error!\n", target.c_str());
    }
}

BatchNorm1d::BatchNorm1d(std::string prefix, size_t num_features, float eps, float monmentum, bool affine, bool track_running_stats) {
    this->num_features = num_features;
    this->eps = eps;
    this->momentum = monmentum;
    this->affine = affine;
    this->track_running_stats = track_running_stats;

    this->prefix = prefix;

    if(affine) {
        this->weights = new Tensor({num_features}, ONES);
        this->bias = new Tensor({num_features}, ZERO);
    } else {
        // the output is just centralization
        this->weights = nullptr;
        this->bias = nullptr;
    }

    if(this->track_running_stats) {
        this->running_mean = new Tensor({num_features}, ZERO);
        this->running_var = new Tensor({num_features}, ONES);
    } else {
        // use batch statistics(bias estimate)
        this->running_mean = nullptr;
        this->running_var = nullptr;
    }
}

BatchNorm1d::BatchNorm1d(size_t num_features, float eps, float monmentum, bool affine, bool track_running_stats) {
    this->num_features = num_features;
    this->eps = eps;
    this->momentum = monmentum;
    this->affine = affine;
    this->track_running_stats = track_running_stats;

    if(affine) {
        this->weights = new Tensor({num_features}, ONES);
        this->bias = new Tensor({num_features}, ZERO);
    } else {
        // the output is just centralization
        this->weights = nullptr;
        this->bias = nullptr;
    }

    if(this->track_running_stats) {
        this->running_mean = new Tensor({num_features}, ZERO);
        this->running_var = new Tensor({num_features}, ONES);
    } else {
        // use batch statistics(bias estimate)
        this->running_mean = nullptr;
        this->running_var = nullptr;
    }
}

BatchNorm1d::~BatchNorm1d() {
    delete running_mean, running_var, weights, bias, input, output, outputBackward;
}

Tensor* BatchNorm1d::forward(Tensor* data) {
    if(!this->track_running_stats){
        // Dependency: Tensor.mean(size_t dim);
        ERROR("not implemented!");
    }
    this->reset();
    if(this->is_training) {
        this->input = data;
    }

    size_t dim = data->getDim();
    if(dim == 2) {
        DimVector shape = data->getShape();
        int N = shape[0], C = shape[1];
        this->output = new Tensor(shape);

        dim3 block(BLOCK_SIZE2D, BLOCK_SIZE2D);
        dim3 grid((C-1)/BLOCK_SIZE2D + 1, (N-1)/BLOCK_SIZE2D+1); 

        kBn1d_l2<<<grid, block>>>(data->getData(), output->getData(), weights->getData(), bias->getData(), running_mean->getData(), running_var->getData(), eps, N, C); CHECK_KERNEL();
    } else if(dim == 3) {
        DimVector shape = data->getShape();
        int N = shape[0], C = shape[1], L = shape[2];
        this->output = new Tensor(shape);

        dim3 block(BLOCK_SIZE3D, BLOCK_SIZE3D, BLOCK_SIZE3D);
        dim3 grid((L-1)/BLOCK_SIZE3D + 1, (C-1)/BLOCK_SIZE3D+1, (N-1)/BLOCK_SIZE3D+1); 

        kBn1d_l3<<<grid, block>>>(data->getData(), this->output->getData(), weights->getData(), bias->getData(), running_mean->getData(), running_var->getData(), eps, N, C, L); CHECK_KERNEL();
    } else {
        ERROR("Dimension not allowed!");
    }
    return this->output;
}


Tensor* BatchNorm1d::backward(Tensor* gradients){
    return nullptr;
}