#include "bn1d.cuh"

/* [(N x C) - (C)] / sqrt(C + eps) * (C) + (C) */
__global__ 
void kBn1d_l2(
    float* d_data, float* d_out, 
    float* weights, float* bias, float* mean, float* var, 
    float eps, int N, int C, bool relu
) {
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;

    if(row >= N || col >= C) return;

    float scaling = rsqrtf(var[col] + eps); // 1 / sqrt(C + eps)
    float norm = (d_data[row * C + col] - mean[col]) * scaling;

    float cVal = norm * weights[col] + bias[col];
    d_out[row * C + col] = relu ? fmaxf(cVal, 0.0) : cVal;
}

__global__ 
void kBn1d_cache_l2(
    float* d_data, float* d_out, 
    float* weights, float* bias, float* mean, float* var, 
    float* x_hat, float* x_minus_mu, float* sqrt_var_inv,
    float eps, int N, int C, bool relu
) {
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;

    if(row >= N || col >= C) return;
    int offset2d = row*C + col;

    float scaling = rsqrtf(var[col] + eps); // 1 / sqrt(C + eps)
    float x_m = d_data[offset2d] - mean[col];
    float norm = x_m * scaling;

    float cVal = norm * weights[col] + bias[col];
    d_out[offset2d] = relu ? fmaxf(cVal, 0.0) : cVal;

    // cache
    x_hat[offset2d] = norm;
    x_minus_mu[offset2d] = x_m;
    if(threadIdx.y == 0)
        sqrt_var_inv[col] = scaling;
}


/* [(N x C x L) - (C)] / sqrt(C + eps) * (C) + (C) */
__global__ 
void kBn1d_l3(
    float* d_data, float* d_out, 
    float* weights, float* bias, float* mean, float* var,
    float eps, int N, int C, int L, bool relu
) {
    int z = threadIdx.z + blockIdx.z * blockDim.z;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int x = threadIdx.x + blockIdx.x * blockDim.x;

    if(z >= N || y >= C || x >= L) return;

    float scaling = rsqrtf(var[y] + eps); // 1 / sqrt(C + eps)
    float norm = (d_data[z*C*L + y*L + x] - mean[y]) * scaling;

    float cVal = norm * weights[y] + bias[y];
    d_out[z*C*L + y*L + x] = relu ? fmaxf(cVal, 0.0) : cVal;
}
/* [(N x C x L) - (C)] / sqrt(C + eps) * (C) + (C) */
__global__ 
void kBn1d_cache_l3(
    float* d_data, float* d_out, 
    float* weights, float* bias, float* mean, float* var, 
    float* x_hat, float* x_minus_mu, float* sqrt_var_inv,
    float eps, int N, int C, int L, bool relu
) {
    int z = threadIdx.z + blockIdx.z * blockDim.z;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int x = threadIdx.x + blockIdx.x * blockDim.x;

    if(z >= N || y >= C || x >= L) return;

    int offset3d = z*C*L + y*L + x;

    float scaling = rsqrtf(var[y] + eps); // 1 / sqrt(C + eps)
    
    float x_m = d_data[offset3d] - mean[y];
    float norm = x_m * scaling;
    

    float cVal = norm * weights[y] + bias[y];
    if(cVal > 1000) {
        printf("x_n = %f, %f = 1/(%f + %f), weight = %f, bias = %f\n", x_m, scaling, var[y], eps, weights[y], bias[y]);
    }
    d_out[offset3d] = relu ? fmaxf(cVal, 0.0) : cVal;

    // cache
    x_hat[offset3d] = norm;
    x_minus_mu[offset3d] = x_m;
    if(threadIdx.x == 0 && threadIdx.z == 0)
        sqrt_var_inv[y] = scaling;
}

__global__
void kBackprop_to_mean_and_var_l2(
    float* d_mean, float* d_var, float* d_x_hat, float* sqrt_var_inv, float* x_minus_mu, int N, int C, int stride
) {
    int row = threadIdx.y;  // single row of width: BATCH_BASE
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    
    // float d_m = 0.0f, d_v = 0.0f;
    __shared__ float ds_var[BLOCK_SIZE1D][BATCH_BASE];
    __shared__ float ds_mean[BLOCK_SIZE1D][BATCH_BASE];

    if(col >= C) return;

    int iter = (N-1)/BATCH_BASE + 1;
    for(int i=0; i<iter; i++) {
        int real_row = (row + i*BATCH_BASE);
        int idx2d = real_row * C + col;
        int idx1d = col / stride;
        int tid = threadIdx.y;

        if(real_row < N) {
            float tmp = - d_x_hat[idx2d] * sqrt_var_inv[idx1d];
            ds_mean[threadIdx.x][threadIdx.y] = tmp;
            ds_var[threadIdx.x][threadIdx.y] = 0.5 * tmp * x_minus_mu[idx2d] * powf(sqrt_var_inv[idx1d], 2);
            __syncthreads();
        }
            

        for(int stride=BATCH_BASE/2; stride>0; stride>>=1) {
            if(tid < stride && tid + stride + i*BATCH_BASE < N) {
                ds_mean[threadIdx.x][tid] = ds_mean[threadIdx.x][tid] + ds_mean[threadIdx.x][tid + stride];
                ds_var[threadIdx.x][tid] = ds_var[threadIdx.x][tid] + ds_var[threadIdx.x][tid + stride];
            }
            __syncthreads();
        }

        if(threadIdx.y == 0 && col < C && real_row < N) {
            d_mean[idx1d] = ds_mean[threadIdx.x][0];
            d_var[idx1d] = ds_var[threadIdx.x][0];
        }
    }
}

__global__
void kBn1d_back_l2(
    float* d_in, 
    float* d_x_hat, float* d_var, float* d_mean,
    float* sqrt_var_inv, float* x_minus_mu, int N, int C
) {
    int col = threadIdx.x + blockDim.x * blockIdx.x;
    int row = threadIdx.y + blockDim.y * blockIdx.y;

    if(col >= C || row >= N) return;

    int idx2d = row * C + col;
    float f1 = sqrt_var_inv[col];
    float f2 = 2 * x_minus_mu[idx2d] / N;

    d_in[idx2d] = d_x_hat[idx2d] * f1 + d_var[col] * f2 + d_mean[col] / N;
}

__global__
void kBn1d_back_l3(
    float* d_in, 
    float* d_x_hat, float* d_var, float* d_mean,
    float* sqrt_var_inv, float* x_minus_mu, int N, int C, int L
) {
    int col = threadIdx.x + blockDim.x * blockIdx.x;
    int row = threadIdx.y + blockDim.y * blockIdx.y;
    int batch = threadIdx.z + blockDim.z * blockIdx.z;

    if(col >= L || row >= C || batch >= N) return;

    int idx3d = batch * C * L + row * L + col;

    float f1 = sqrt_var_inv[row];
    float f2 = 2 * x_minus_mu[idx3d] / N;

    d_in[idx3d] = d_x_hat[idx3d] * f1 + d_var[row] * f2 + d_mean[row] / N;
}

__global__
void kBackprop_to_mean_and_var_l3(
    float* d_mean, float* d_var, float* d_x_hat, float* sqrt_var_inv, float* x_minus_mu, int N, int C, int L
) {
    
    int row = threadIdx.y;  // single row of width: BATCH_BASE
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    
    // float d_m = 0.0f, d_v = 0.0f;
    __shared__ float ds_var[BLOCK_SIZE1D][BATCH_BASE];
    __shared__ float ds_mean[BLOCK_SIZE1D][BATCH_BASE];

    if(col >= C) return;

    int iter = (N-1)/BATCH_BASE + 1;

    for(int i=0; i<iter; i++) {
        int real_row = (row + i*BATCH_BASE);
        int idx2d = real_row * C + col;
        int idx1d = col;
        int tid = threadIdx.y;

        float tmp = - d_x_hat[idx2d] * sqrt_var_inv[idx1d];
        
        ds_mean[threadIdx.x][threadIdx.y] = tmp;
        ds_var[threadIdx.x][threadIdx.y] = 0.5 * tmp * x_minus_mu[idx2d] * powf(sqrt_var_inv[idx1d], 2);

        __syncthreads();

        for(int stride=BATCH_BASE/2; stride>0; stride>>=1) {
            if(tid < stride && tid + stride + i*BATCH_BASE < N) {
                ds_mean[threadIdx.x][tid] = ds_mean[threadIdx.x][tid] + ds_mean[threadIdx.x][tid + stride];
                ds_var[threadIdx.x][tid] = ds_var[threadIdx.x][tid] + ds_var[threadIdx.x][tid + stride];
            }
            __syncthreads();
        }

        if(threadIdx.y == 0 && col < C) {
            d_mean[idx1d] = ds_mean[threadIdx.x][0];
            d_var[idx1d] = ds_var[threadIdx.x][0];
        }
    }
}

__global__
void kBackprop_to_weights_and_bias_l3(
    float* d_mean, float* d_var, float* d_x_hat, float* sqrt_var_inv, float* x_minus_mu, int N, int C, int L
) {
    
}

BatchNorm1d::BatchNorm1d(std::string prefix, size_t num_features, bool relu, float eps, float monmentum, bool affine, bool track_running_stats) {
    this->num_features = num_features;
    this->relu = relu;
    this->eps = eps;
    this->momentum = monmentum;
    this->affine = affine;
    this->track_running_stats = track_running_stats;

    this->prefix = prefix;

    if(affine) {
        this->weights = new Tensor({num_features}, NONE);
        this->bias = new Tensor({num_features}, NONE);
    } else {
        // the output is just centralization
        this->weights = nullptr;
        this->bias = nullptr;
    }

    if(this->track_running_stats) {
        this->running_mean = new Tensor({num_features}, NONE);
        this->running_var = new Tensor({num_features}, NONE);
        if(this->is_training) {
            this->x_minus_mu = new Tensor({num_features}, NONE);
            this->sqrt_var_inv = new Tensor({num_features}, NONE);
        }
    } else {
        // use batch statistics(bias estimate)
        this->running_mean = nullptr;
        this->running_var = nullptr;
    }
}

BatchNorm1d::BatchNorm1d(size_t num_features, bool relu, float eps, float monmentum, bool affine, bool track_running_stats) {
    this->num_features = num_features;
    this->relu = relu;
    this->eps = eps;
    this->momentum = monmentum;
    this->affine = affine;
    this->track_running_stats = track_running_stats;

    if(affine) {
        this->weights = new Tensor({num_features}, NONE);
        this->bias = new Tensor({num_features}, NONE);
    } else {
        // the output is just centralization
        this->weights = nullptr;
        this->bias = nullptr;
    }

    if(this->track_running_stats) {
        this->running_mean = new Tensor({num_features}, NONE);
        this->running_var = new Tensor({num_features}, NONE);
        if(this->is_training) {
            this->x_minus_mu = new Tensor({num_features}, NONE);
            this->sqrt_var_inv = new Tensor({num_features}, NONE);
        }
    } else {
        // use batch statistics(bias estimate)
        this->running_mean = nullptr;
        this->running_var = nullptr;
    }
}

BatchNorm1d::~BatchNorm1d() {
    if(this->track_running_stats) {
        delete running_mean;
        delete running_var;
    }
    if(x_minus_mu!= nullptr) delete x_minus_mu;
    if(sqrt_var_inv != nullptr) delete sqrt_var_inv;
    if(x_hat!= nullptr) delete x_hat;
}

BatchNorm1d* BatchNorm1d::train() {
    BaseLayer::train();
    this->is_training = true;
    size_t bz = Configurer::batch_size;
    size_t l = Configurer::cropping_size;
    if(!d_in) {
        d_in = new Tensor({bz, num_features, l});
    }
    return this;
}

void BatchNorm1d::load_weights() {
    this->weights->fromVec(Configurer::getWeights(this->prefix + "weight"));
    this->bias->fromVec(Configurer::getWeights(this->prefix + "bias"));
    this->running_mean->fromVec(Configurer::getWeights(this->prefix + "running_mean"));
    this->running_var->fromVec(Configurer::getWeights(this->prefix + "running_var"));
}

void BatchNorm1d::init_weights() {
    if(affine) {
        DEBUG_PRINT("BatchNorm1d init weights: ONES\n");
        DEBUG_PRINT("BatchNorm1d init bias: ZERO\n");
        this->weights->initialize(ONES);
        this->bias->initialize(ZERO);
    }

    if(this->track_running_stats) {
        DEBUG_PRINT("BatchNorm1d init mean: ZERO\n");
        DEBUG_PRINT("BatchNorm1d init var: ONES\n");
        this->running_mean->initialize(ZERO);
        this->running_var->initialize(ONES);
    }
}

void BatchNorm1d::load_weights(std::vector<float>& params, const std::string& target) {
    size_t n_data = params.size();
    float* h_data = params.data();
    if(target=="weights") {
        assert(n_data == this->weights->getSize());
        this->weights->load(h_data, n_data);
    } else if(target == "bias") {
        assert(n_data == this->bias->getSize());
        this->bias->load(h_data, n_data);
    } else if(target == "mean") {
        assert(this->running_mean->getSize() == n_data);
        this->running_mean->load(h_data, n_data);
    } else if(target == "var") {
        assert(this->running_var->getSize() == n_data);
        this->running_var->load(h_data, n_data);
    } else {
        ERROR("Load weights %s error!\n", target.c_str());
    }
}

void BatchNorm1d::prepare_backward() {
    // cache for backward
    DimVector shape = input->getShape();
    DimVector point_shape = shape;
    point_shape.erase(point_shape.begin());
    if(!x_hat) {
        x_hat = new Tensor(shape);
    } x_hat->reset(shape);

    if(!x_minus_mu){
        x_minus_mu = new Tensor(shape);
    } x_minus_mu->reset(shape);

    if(!sqrt_var_inv) {
        sqrt_var_inv = new Tensor({num_features});
    } sqrt_var_inv->reset({num_features});

    if(!d_var) {
        d_var = new Tensor({num_features});
    } d_var->reset({num_features});
    if(!d_mean) {
        d_mean = new Tensor({num_features});
    } d_mean->reset({num_features});
}

Tensor* BatchNorm1d::forward(Tensor* data) {
    DEBUG_PRINT("[BatchNorm1d] %sforward\n", this->prefix.c_str());
    if(!this->track_running_stats){
        ERROR("not implemented!");
    }
    DimVector shape_o = data->getShape();

    if(this->output == nullptr) {
        this->output = new Tensor(shape_o);
    } this->output->reset(shape_o);

    if(this->is_training) {
        this->input = data;

        input->mean(mean_cache, 0);
        if(mean_cache->getDim() > 1) {
            mean_cache->mean_(1, true);
        }

        // unroll var function
        input->sub(var_cache, mean_cache);
        var_cache->square_();
        var_cache->mean_(0);
        if(var_cache->getDim() > 1) {
            var_cache->mean_(1);
        }

        // float* mean_tmp = mean_cache->toHost();
        // float* var_tmp = var_cache->toHost();
        // float* input_tmp = input->toHost();
        // printM(input_tmp, input->getShape());
        // printM(mean_tmp, mean_cache->getShape());
        // printM(var_tmp, var_cache->getShape());
        // free(mean_tmp); free(var_tmp); free(input_tmp);

        prepare_backward();
    }


    size_t dim = data->getDim();
    if(dim == 2) {

        DimVector shape = data->getShape();
        int N = shape[0], C = shape[1];

        dim3 block(BLOCK_SIZE1D, BATCH_BASE);
        dim3 grid((C-1)/BLOCK_SIZE1D + 1, (N-1)/BATCH_BASE+1); 

        if(!is_training) {
            kBn1d_l2<<<grid, block>>>(data->getData(), output->getData(), weights->getData(), bias->getData(), running_mean->getData(), running_var->getData(), eps, N, C, this->relu); CHECK_KERNEL();
        } else {
            kBn1d_cache_l2<<<grid, block>>>(data->getData(), output->getData(), weights->getData(), bias->getData(), mean_cache->getData(), var_cache->getData(), x_hat->getData(), x_minus_mu->getData(), sqrt_var_inv->getData(), eps, N, C, this->relu); CHECK_KERNEL();
        }

    } else if(dim == 3) {
        DimVector shape = data->getShape();
        int N = shape[0], C = shape[1], L = shape[2];

        dim3 block(BLOCK_SIZE3D, BLOCK_SIZE3D, BATCH_BASE);
        dim3 grid((L-1)/BLOCK_SIZE3D + 1, (C-1)/BLOCK_SIZE3D+1, (N-1)/BATCH_BASE+1); 
        if(!is_training) {
            kBn1d_l3<<<grid, block>>>(data->getData(), this->output->getData(),  weights->getData(), bias->getData(), running_mean->getData(), running_var->getData(), eps, N, C, L, this->relu); CHECK_KERNEL();
        } else {
            kBn1d_cache_l3<<<grid, block>>>(data->getData(), output->getData(), weights->getData(), bias->getData(), mean_cache->getData(), var_cache->getData(), x_hat->getData(), x_minus_mu->getData(), sqrt_var_inv->getData(), eps, N, C, L, this->relu); CHECK_KERNEL();
        }
    } else {
        ERROR("Dimension not allowed!");
    }
    if(is_training) {
        running_mean->add_(mean_cache->squeeze(), (1 - momentum), momentum);
        running_var->add_(var_cache->squeeze(), (1 - momentum), momentum);
    }

    return this->output;
}


Tensor* BatchNorm1d::backward(Tensor* gradients){
    DEBUG_PRINT("[BatchNorm1d] %sbackward\n", this->prefix.c_str());
    if(this->relu) {
        gradients->mask(output);
    }

    // gradients(N, C, L)
    size_t size = input->getDim();
    DimVector shape = input->getShape();

    // (N, C)/(N, C, L) * (N, C)/(N, C, L) -> sum ->(C)
    gradients->dot(d_weights, x_hat->squeeze());

    d_weights->sumToDim_(1);
    gradients->sumToDim(d_bias, 1);

    // accumulate grads
    weights->acc_grads(d_weights);
    bias->acc_grads(d_bias);

    // backward kernel
    if(size == 2) {
        size_t N = shape[0], C = shape[1];
        this->d_in->reset({N, C});
        gradients->dot(d_x_hat, weights); // (N,C) * (C)
        dim3 block(BLOCK_SIZE1D, BATCH_BASE);
        dim3 grid((C-1)/BLOCK_SIZE1D+1, 1);

        kBackprop_to_mean_and_var_l2<<<grid, block>>>(d_mean->getData(), d_var->getData(), d_x_hat->getData(), sqrt_var_inv->getData(), x_minus_mu->getData(), N, C, 1);CHECK_KERNEL();

        dim3 block_sub(BLOCK_SIZE1D, BATCH_BASE);
        dim3 grid_sub((C-1)/BLOCK_SIZE1D+1, (N-1)/BATCH_BASE+1);

        kBn1d_back_l2<<<grid_sub, block_sub>>>(d_in->getData(), d_x_hat->getData(), d_var->getData(), d_mean->getData(), sqrt_var_inv->getData(), x_minus_mu->getData(), N, C);CHECK_KERNEL();
    } else if(size==3) {
        size_t N = shape[0], C = shape[1], L = shape[2];
        this->d_in->reset({N, C, L});
        // DEBUG_PRINT("(%d, %d, %d)\n", N, C, L);
        weights->unsqueeze(-1);
        gradients->dot(d_x_hat, weights); //(N,C,L)*(C,1)
        weights->squeeze(-1);

        dim3 block(BLOCK_SIZE1D, BATCH_BASE);
        dim3 grid((C*L-1)/BLOCK_SIZE1D+1, 1);

        // printShape(d_mean->getShape());         // (C)
        // printShape(d_var->getShape());          // (C)
        // printShape(sqrt_var_inv->getShape());   // (C)
        // printShape(d_x_hat->getShape());        // (N, C, L)
        // printShape(x_minus_mu->getShape());     // (N, C, L)

        kBackprop_to_mean_and_var_l2<<<grid, block>>>(d_mean->getData(), d_var->getData(), d_x_hat->getData(), sqrt_var_inv->getData(), x_minus_mu->getData(), N, C*L, L);CHECK_KERNEL();

        dim3 block_sub(BLOCK_SIZE2D, BLOCK_SIZE2D, BATCH_BASE);
        dim3 grid_sub((L-1)/BLOCK_SIZE2D+1, (C-1)/BLOCK_SIZE1D+1, (N-1)/BATCH_BASE+1);
        kBn1d_back_l3<<<grid_sub, block_sub>>>(d_in->getData(), d_x_hat->getData(), d_var->getData(), d_mean->getData(), sqrt_var_inv->getData(), x_minus_mu->getData(), N, C, L);CHECK_KERNEL();
    } else {
        ERROR("Size not matched!\n");
    }

    return d_in;
}
