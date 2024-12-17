#include "tensor.cuh"

// Declarations
bool isContinuousSqueeze(DimVector& shape1, DimVector& shape2);
DimVector getBroadcastShape(DimVector& shape1, DimVector& shape2, DimVector& stride1, DimVector& stride2);
bool checkMatmulShape(DimVector& shape1, DimVector& shape2);
DimVector getMatmulShape(DimVector& shape1, DimVector& shape2);
DimVector getBatchMatmulShape(DimVector& shape1, DimVector& shape2);

Tensor::Tensor(DimVector shape, InitType init_type) {
    this->shape = shape;
    int dim = shape.size();
    this->n_data = accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
    CHECK(cudaMalloc((float**)&(this->d_data), this->n_data * sizeof(float)));
    this->initialize(init_type);
}

Tensor::Tensor(const Tensor &other) {
    this->shape = other.shape;
    this->n_data = other.n_data;
    CHECK(cudaMalloc(&(this->d_data), n_data * sizeof(float)));
    CHECK(cudaMemcpy(this->d_data, other.d_data, this->n_data * sizeof(float), cudaMemcpyDeviceToDevice));
}

Tensor::Tensor(std::vector<float>& data_vec, DimVector shape) {
    this->shape = shape;
    this->n_data = data_vec.size();
    CHECK(cudaMalloc(&(this->d_data), n_data * sizeof(float)));
    CHECK(cudaMemcpy(this->d_data, (float*) data_vec.data(), n_data * sizeof(float), cudaMemcpyHostToDevice));
}

void Tensor::copyFrom(Tensor* other) {
    this->shape = other->shape;
    this->reset(shape);
}

bool Tensor::checkNan(bool check_grad) {
    // DEBUG_PRINT("[HOOK] checkNan()\n");
    size_t num = this->getSize();
    bool* d_result;
    bool h_result = false;
    CHECK(cudaMalloc(&d_result, sizeof(bool)));
    CHECK(cudaMemcpy(d_result, &h_result, sizeof(bool), cudaMemcpyHostToDevice));

    int blockSize = BLOCK_SIZE1D;
    int gridSize = (num - 1) / blockSize+1;

    kCheckNaN<<<gridSize, blockSize>>>(d_data, num, d_result); CHECK_KERNEL();
    cudaMemcpy(&h_result, d_result, sizeof(bool), cudaMemcpyDeviceToHost);

    CHECK(cudaFree(d_result));

    return h_result;
}

Tensor::~Tensor(){
    CHECK(cudaFree(this->d_data));

    if(this->transpose_cache != nullptr) {
        CHECK(cudaFree(this->transpose_cache));
    }

    if(!this->gradients_acc) {
        delete gradients_acc;
    }
}

bool Tensor::is_train() {
    return this->is_training;
}

void Tensor::train() {
    this->is_training = true;
    if(!this->gradients_acc) {
        this->gradients_acc = new Tensor(this->shape, ZERO);
    } this->gradients_acc->reset(this->shape);
}

void Tensor::eval() {
    this->is_training = false;
    if(this->gradients_acc) {
        delete gradients_acc;
        gradients_acc = nullptr;
    }
}

void Tensor::acc_grads(Tensor* grads) {
    if(!is_training) return;
    if(this->gradients_acc == nullptr) {
        ERROR("Training Tensor w/o gradients!\n");
    }
    if(this->shape != grads->shape) {
        printShape(this->shape);
        printShape(grads->shape);
        ERROR("gradients not matched!\n");
    }
    this->gradients_acc->add_(grads);
}

Tensor* Tensor::getGradsAcc(){
    return this->gradients_acc;
}

void Tensor::reset(DimVector shape) {
    size_t size = accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
    if(this->getMemSize() >= size) {
        this->shape = shape;
    } else {
        DEBUG_PRINT("%ld < %ld, tensor has been reset\n", this->getMemSize(), size);
        this->n_data = size;
        this->shape = shape;
        CHECK(cudaFree(this->d_data));
        CHECK(cudaMalloc((float**)(&this->d_data), size * sizeof(float)));
    }
}

void Tensor::fromVec(std::vector<float>& vec) {
    if(vec.size() > this->n_data) {
        ERROR("%ld != %ld, weight size not matched!\n", vec.size(), this->n_data);
    }
    float* h_data = vec.data();
    CHECK(cudaMemcpy(this->d_data, h_data, vec.size() * sizeof(float), cudaMemcpyHostToDevice));
}

std::vector<float> Tensor::toVec() {
    float* h_data = this->toHost();
    size_t size = this->getSize();
    std::vector<float> vec(h_data, h_data + size);
    return vec;
}

/* getters */

size_t Tensor::getDim() {
    return this->shape.size();
}

size_t Tensor::getSize() {
    size_t size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());

    return size;
}

size_t Tensor::getMemSize() {
    return this->n_data;
}

DimVector Tensor::getShape(){
    return this->shape;
}

size_t Tensor::getSize(int dim) {
    if(dim < 0) dim += shape.size();
    return this->shape.at(dim);
}

float* Tensor::toHost(){
    size_t nBytes = this->getSize() * sizeof(float);
    float* h_d = (float *) malloc(nBytes);
    CHECK(cudaMemcpy(h_d, d_data, nBytes, cudaMemcpyDeviceToHost));
    return h_d;
}

float* Tensor::getData() {
    return this->d_data;
}

/* setters */

void Tensor::setData(float* data) {
    this->d_data = data;
}

void Tensor::load(float* h_data, size_t n_data) {
    assert(n_data = this->n_data);
    CHECK(cudaMemcpy(d_data, h_data, n_data * sizeof(float), cudaMemcpyHostToDevice));
}

void Tensor::initialize(float value) {
    int block = BLOCK_SIZE1D;
    size_t size = this->getSize();
    int grid = (size - 1) / BLOCK_SIZE1D + 1;
    kFill<<<grid, block>>>(d_data, value, size); CHECK_KERNEL();
}

void Tensor::initialize(InitType type, float bound) {
    if(type==NONE) return;
    
    size_t size = this->getSize();
    size_t nBytes = this->getSize() * sizeof(float);
    float* h_d = (float *)malloc(nBytes);
    int block = BLOCK_SIZE1D;
    int grid = (size - 1) / BLOCK_SIZE1D + 1;

    if(type == ZERO) {
        kFill<<<grid, block>>>(d_data, 0.0f, size); CHECK_KERNEL();
    } else if(type==ONES) {
        kFill<<<grid, block>>>(d_data, 1.0f, size); CHECK_KERNEL();
    } else if(type==IDENTITY) {
        assert(shape.size() == 2 && shape[0] == shape[1]);
        int s = shape[0];
        for (int i=0; i < s; i++) {
            for(int j=0; j < s; j++) {
                if(i==j) h_d[i*s + j] = 1.0f;
                else     h_d[i*s + j] = 0.0f;
            }
        }
        CHECK(cudaMemcpy(d_data, h_d, nBytes, cudaMemcpyHostToDevice));
    } else if(type==RANDOM) {
        float MAX = 1.0f;
        float MIN = -1.0f;
        randomFloatMatrix(h_d, getSize(), MIN, MAX);
        CHECK(cudaMemcpy(d_data, h_d, nBytes, cudaMemcpyHostToDevice));
    } else if(type==KAIMING) {
        // saved for Kaiming uniform
        float MAX = bound;
        float MIN = -bound;
        randomFloatMatrix(h_d, getSize(), MIN, MAX);
        CHECK(cudaMemcpy(d_data, h_d, nBytes, cudaMemcpyHostToDevice));
    }else {
        ERROR("Not implemented!");
    }
    
    free(h_d);
}

void Tensor::initialize(float *h_data, DimVector& shape) {
    if(this->shape != shape) ERROR("Shape not matched!\n");
    size_t n_data = 1;
    for(int i=0; i<shape.size(); i++) {
        n_data *= shape[i];
    }
    size_t nBytes = n_data * sizeof(float);
    CHECK(cudaMemcpy(d_data, h_data, nBytes, cudaMemcpyHostToDevice));
}

/* Unary op */

/*
3 types reshape:
 - swap dim
 - squeeze dim
 - unsqueeze dim
only continuous compression are supported (just change the shape).
*/
void Tensor::reshape(DimVector shape_n) {
    size_t dim = this->getDim();
    size_t dim_n = shape_n.size();

    if(shape_n == this->shape) return;

    if(dim_n == dim) {
        printShape(this->shape);
        printShape(shape_n);
        // dim swap, internal transpose?
        // TODO
        ERROR("Not implemented!\n");
    } else {
        if(isContinuousSqueeze(this->shape, shape_n)) {
            this->shape = shape_n;
        } else {
            ERROR("Failed to reshape!\n");
        }
    }
}

void Tensor::flatten() {
    this->shape = {accumulate(shape.begin(), shape.end(), (size_t)1, std::multiplies<size_t>())};
}

void Tensor::transpose() {
    if(this->shape.size() == 2) {
        DimVector shape_o = this->shape;
        std::swap(shape_o[0], shape_o[1]);

        if(this->transpose_cache == nullptr) {
            CHECK(cudaMalloc((float**)&transpose_cache, this->getSize() * sizeof(float)));
        }

        int row = this->shape[0], col = this->shape[1];
        dim3 block(BLOCK_SIZE2D, BLOCK_SIZE2D);
        dim3 grid((col-1)/BLOCK_SIZE2D + 1, (row-1)/BLOCK_SIZE2D + 1);

        kTranspose<<<grid, block>>>(this->d_data, transpose_cache, row, col); CHECK_KERNEL();

        float* tmp = this->d_data;
        this->d_data = transpose_cache;
        this->transpose_cache = tmp;
        this->shape = shape_o;
    } else {
        printShape(this->shape);
        ERROR("Transpose failed: the dim != 2.\n");
    }
}

void Tensor::transpose(int dim1, int dim2) {
    size_t dim = this->getDim();
    if(dim1 < 0) dim1 = dim + dim1;
    if(dim2 < 0) dim2 = dim + dim2;
    int t = dim1 <= dim2 ? dim1 : dim2;
    dim2 = dim1 <= dim2 ? dim2 : dim1;
    dim1 = t;

    if(dim < 2 || dim2 - dim1 != 1) {
        ERROR("Failed to transpose(%d, %d)\n", dim1, dim2);
    }

    if(dim == 2) {
        this->transpose();
    } else if(dim == 3) {
        if(dim2 == dim - 1) {
            DimVector shape_o = this->shape;
            std::swap(shape_o[1], shape_o[2]);

            if(this->transpose_cache == nullptr) {
                CHECK(cudaMalloc((float**)&transpose_cache, this->getSize() * sizeof(float)));
            }

            size_t N = shape[0], row = shape[1], col = shape[2];
            dim3 block(BLOCK_SIZE2D, BLOCK_SIZE2D, BATCH_BASE);
            dim3 grid((col-1)/BLOCK_SIZE2D + 1, (row-1)/BLOCK_SIZE2D+1, (N-1)/BATCH_BASE + 1);

            kTransposeLast3D<<<grid, block>>>(this->d_data, this->transpose_cache, N, row, col); CHECK_KERNEL();

            float* tmp = this->d_data;
            this->d_data = transpose_cache;
            this->transpose_cache = tmp;
            this->shape = shape_o;
        } else {
            ERROR("dim1=%d, dim2=%d, Not implemented!\n", dim1, dim2);
        }
    }
}

Tensor* Tensor::scale_(float factor) {
    dim3 block(BLOCK_SIZE1D);
    dim3 grid((this->getSize()-1)/BLOCK_SIZE1D + 1);
    // need padding?
    kScale<<<grid, block>>>(d_data, factor, 0.0f, this->getSize());
    CHECK_KERNEL();
    return this;
}

void Tensor::mask_fill_(Tensor*& mask, int dim, float value) {
    if(this->shape[0] != mask->getSize(0)) {
        ERROR("Batch size not matched: %ld != %ld\n", this->shape[0], mask->getSize(0));
    }

    if(dim < 0) dim += this->getDim();

    if(dim == 2) {  // last dimension
        // int* d_mask;
        int N = shape[0], C=shape[1], L=shape[2];

        dim3 block(BLOCK_SIZE3D, BLOCK_SIZE3D, BATCH_BASE);
        dim3 grid((L-1)/BLOCK_SIZE3D + 1, (C-1)/BLOCK_SIZE3D+1, (N-1)/BATCH_BASE+1);

        kMaskFillLast3D<<<grid, block>>>(this->d_data, mask->getData(), value, N, C, L); CHECK_KERNEL();
    } else {
        ERROR("Not implemented!\n");
    }
}

Tensor* Tensor::exp() {
    DimVector shape_o = this->shape;
    Tensor* tensor_o = new Tensor(shape_o);
    int n_data = this->getSize();
    int block = BLOCK_SIZE1D;
    int grid = (n_data - 1)/BLOCK_SIZE1D + 1;

    kExp<<<grid, block>>>(this->d_data, tensor_o->d_data, n_data); CHECK_KERNEL();
    return tensor_o;
}

void Tensor::exp_() {
    int n_data = this->getSize();
    int block = BLOCK_SIZE1D;
    int grid = (n_data - 1)/BLOCK_SIZE1D + 1;

    kExp<<<grid, block>>>(this->d_data, this->d_data, n_data); CHECK_KERNEL();
}

Tensor* Tensor::log() {
    DimVector shape_o = this->shape;
    Tensor* tensor_o = new Tensor(shape_o);
    int n_data = this->getSize();
    int block = BLOCK_SIZE1D;
    int grid = (n_data - 1)/BLOCK_SIZE1D + 1;

    kLog<<<grid, block>>>(this->d_data, tensor_o->d_data, n_data); CHECK_KERNEL();
    return tensor_o;
}

void Tensor::log_() {
    int n_data = this->getSize();
    int block = BLOCK_SIZE1D;
    int grid = (n_data - 1)/BLOCK_SIZE1D + 1;

    kLog<<<grid, block>>>(this->d_data, this->d_data, n_data); CHECK_KERNEL();
}

void Tensor::square_() {
    int n_data = this->getSize();
    int block = BLOCK_SIZE1D;
    int grid = (n_data - 1)/BLOCK_SIZE1D + 1;

    kSquare<<<grid, block>>>(this->d_data, this->d_data, n_data); CHECK_KERNEL();
}

void Tensor::add_(float c) {
    dim3 block(BLOCK_SIZE1D);
    dim3 grid((this->getSize()-1)/BLOCK_SIZE1D + 1);
    // need padding?
    kScale<<<grid, block>>>(this->d_data, 1.0f, c, this->getSize());
    CHECK_KERNEL();
}

void Tensor::sub_(float c) {
    dim3 block(BLOCK_SIZE1D);
    dim3 grid((this->getSize()-1)/BLOCK_SIZE1D + 1);
    // need padding?
    kScale<<<grid, block>>>(this->d_data, 1.0f, -c, this->getSize());
    CHECK_KERNEL();
}

float Tensor::sum(){
    int block_num = (this->getSize()-1)/BLOCK_SIZE1D + 1;
    float* d_out;
    CHECK(cudaMalloc((float**)&d_out, block_num * sizeof(float)));

    kSum<<<block_num, BLOCK_SIZE1D>>>(this->d_data, d_out, this->getSize());CHECK_KERNEL();

    float* h_out = (float *)malloc(sizeof(float));
    CHECK(cudaMemcpy(h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_out));
    DEBUG_PRINT("memcpy 11\n");
    float tmp = h_out[0]; free(h_out);
    return tmp;
}

Tensor* Tensor::sumToDim(Tensor*& tensor_o, int dim){
    size_t size = this->getDim();
    if(dim < 0) {
        dim += size;
    }

    if(size == 2 && dim == 1) {
        size_t N = shape[0], C = shape[1];
        DimVector shape_o = {C};
        if(tensor_o== nullptr) {
            tensor_o = new Tensor(shape_o);
        } tensor_o->reset(shape_o);

        dim3 block(BLOCK_SIZE1D, BATCH_BASE);
        dim3 grid((C-1)/BLOCK_SIZE1D + 1);
        kBatchReduce2D<<<grid, block>>>(this->d_data, tensor_o->d_data, N, C, false);CHECK_KERNEL();
    } else if(size==2 && dim == 0) {
        size_t N = shape[0], C = shape[1];
        DimVector shape_o = {N};
        if(tensor_o== nullptr) {
            tensor_o = new Tensor(shape_o);
        } tensor_o->reset(shape_o);

        dim3 block(BLOCK_SIZE1D);
        dim3 grid(N);
        kSumLastDim2D<<<grid, block>>>(tensor_o->getData(), tensor_o->getData(), N, C, false);CHECK_KERNEL();
    } else if(size == 3 && dim == 1) {
        size_t N = shape[0], C = shape[1], L = shape[2];
        DimVector shape_o = {C, L};
        if(tensor_o== nullptr) {
            tensor_o = new Tensor(shape_o);
        } tensor_o->reset(shape_o);

        dim3 block(BLOCK_SIZE1D, BATCH_BASE);
        dim3 grid((C*L-1)/BLOCK_SIZE1D + 1, 1);
        kBatchReduce2D<<<grid, block>>>(this->getData(), tensor_o->getData(), N, C*L, false);CHECK_KERNEL();

        int block_sub = BLOCK_SIZE1D;
        int grid_sub = C;
        kSumLastDim2D<<<grid_sub, block_sub>>>(tensor_o->getData(), tensor_o->getData(), C, L, false);CHECK_KERNEL();
        tensor_o->reset({C});
    } else if (size == 1 && dim == 0) {
        return this;    // maybe bug?
    } else {
        ERROR("size = %ld, dim = %d, not implemented\n", size, dim);
    }
    return tensor_o;
}

Tensor* Tensor::mask(Tensor* ref) {
    if(shape != ref->getShape()) {

        printShape(this->getShape());
        printShape(ref->getShape());
        ERROR("mask tensor shape not matched!\n");
    }
    size_t size = shape.size();
    if(size == 2) {
        int N=shape[0], C=shape[1];
        dim3 block(BLOCK_SIZE1D, BATCH_BASE);
        dim3 grid((C-1)/BLOCK_SIZE1D+1, (N-1)/BATCH_BASE+1);
        kMask_l2<<<grid, block>>>(this->getData(), ref->getData(), N, C); CHECK_KERNEL();
    } else if (size == 3) {
        int N=shape[0], C=shape[1], L=shape[2];
        dim3 block(BLOCK_SIZE2D, BLOCK_SIZE2D, BATCH_BASE);
        dim3 grid((L-1)/BLOCK_SIZE2D+1, (C-1)/BLOCK_SIZE2D+1, (N-1)/BATCH_BASE+1);
        kMask_l3<<<grid, block>>>(this->getData(), ref->getData(), N, C, L); CHECK_KERNEL();
    } else {
        ERROR("Not implemented!\n");
    }
    return this;
}

Tensor* Tensor::sumToDim_(int dim){
    int size = this->getDim();
    if(dim < 0) {
        dim += size;
    }

    if(size == 1 && dim == 0) {
        return this;
    }

    if(size == 2 && dim == 1) {
        size_t N = shape[0], C = shape[1];
        DimVector shape_o = {C};
        dim3 block(BLOCK_SIZE1D, BATCH_BASE);
        dim3 grid((C-1)/BLOCK_SIZE1D + 1, 1);
        kBatchReduce2D<<<grid, block>>>(d_data, d_data, N, C, false);CHECK_KERNEL();
        this->shape = shape_o;
    } else if(size==2 && dim == 0) {

        size_t N = shape[0], C = shape[1];
        DimVector shape_o = {N};

        dim3 block(BLOCK_SIZE1D, BATCH_BASE);
        dim3 grid(1, (N-1)/BATCH_BASE+1);

        DEBUG_PRINT("in the sumToDim\n");
        kSumLastDim2D<<<grid, block>>>(d_data, d_data, N, C, false);CHECK_KERNEL();
        DEBUG_PRINT("out the sumToDim\n");

        this->shape = shape_o;
    } else if(size == 3 && dim == 1) {
        size_t N = shape[0], C = shape[1], L = shape[2];
        DimVector shape_o = {C, L};
        dim3 block(BLOCK_SIZE1D, BATCH_BASE);
        dim3 grid((C*L-1)/BLOCK_SIZE1D + 1);
        kBatchReduce2D<<<grid, block>>>(d_data, d_data, N, C*L, false);CHECK_KERNEL();

        int block_sub = BLOCK_SIZE1D;
        int grid_sub = C;
        kSumLastDim2D<<<grid_sub, block_sub>>>(d_data, d_data, C, L, false);CHECK_KERNEL();
        this->reset({C});
    } else {
        ERROR("size=%d, dim=%d, Not implemented\n", size, dim);
    }
    return this;
}

float Tensor::mean(){
    float s = sum();
    return s/this->getSize();
}

Tensor* Tensor::mean(Tensor*& tensor_o, int dim, bool keepDim){
    int size = this->shape.size();
    if(dim < 0) {
        dim = dim + this->getDim();
    }
    int dim_size = this->getSize(dim);
    // check the shape_o
    DimVector shape_o = this->shape;
    shape_o[dim] = 1; 
    // reset the tensor_o
    if(tensor_o == nullptr) {
        tensor_o = new Tensor(shape_o);
    } 
    tensor_o->reset(shape_o);
    if(!keepDim) tensor_o->squeeze(dim);

    if(size == 2 && dim == 0) {
        size_t N = shape[0], C = shape[1];
        dim3 block(BLOCK_SIZE1D, BATCH_BASE);
        dim3 grid((C-1)/BLOCK_SIZE1D + 1);
        kBatchReduce2D<<<grid, block>>>(this->d_data, tensor_o->d_data, N, C, true);CHECK_KERNEL();
    } else if(size == 3 && dim == 0) {
        size_t N = shape[0], C = shape[1], L=shape[2];
        dim3 block(BLOCK_SIZE1D, BATCH_BASE);
        dim3 grid((C*L -1)/BLOCK_SIZE1D + 1);
        kBatchReduce2D<<<grid, block>>>(this->d_data, tensor_o->d_data, N, C*L, true);
    } else if(size == 2 && dim == 1) {
        size_t N = shape[0], C = shape[1];
        dim3 block(BLOCK_SIZE1D, BATCH_BASE);
        dim3 grid(1, (N-1)/BATCH_BASE + 1);
        kSumLastDim2D<<<grid, block>>>(this->d_data, tensor_o->d_data, N, C, true);
    } else {
        ERROR("Not implemented!\n");
    }
    return tensor_o;
}

Tensor* Tensor::mean_(int dim, bool keepDim){
    int size = this->shape.size();
    if(dim < 0) {
        dim = dim + this->getDim();
    }
    int dim_size = this->getSize(dim);

    if(size == 2 && dim == 0) {
        size_t N = shape[0], C = shape[1];
        dim3 block(BLOCK_SIZE1D, BATCH_BASE);
        dim3 grid((C-1)/BLOCK_SIZE1D + 1);
        kBatchReduce2D<<<grid, block>>>(this->d_data, this->d_data, N, C, true);CHECK_KERNEL();
    } else if(size == 3 && dim == 0) {
        size_t N = shape[0], C = shape[1], L=shape[2];
        dim3 block(BLOCK_SIZE1D, BATCH_BASE);
        dim3 grid((C*L -1)/BLOCK_SIZE1D + 1);
        kBatchReduce2D<<<grid, block>>>(this->d_data, this->d_data, N, C*L, true);
    } else if(size == 2 && dim == 1) {
        size_t N = shape[0], C = shape[1];
        dim3 block(BLOCK_SIZE1D, BATCH_BASE);
        dim3 grid(1, (N-1)/BATCH_BASE + 1);
        kSumLastDim2D<<<grid, block>>>(this->d_data, this->d_data, N, C, true);
    } else {
        ERROR("Not implemented!\n");
    }
    this->shape[dim] = 1;
    if(!keepDim) this->squeeze(dim);
    return this;
}

/* Theory BUG !!! */
Tensor* Tensor::var(Tensor*& tensor_o, int dim, Tensor* mean, bool keepDim){
    if(mean == nullptr) {
        ERROR("Not implemented\n");
    }
    
    int size = this->shape.size();
    if(dim < 0) {
        dim = dim + this->getDim();
    }
    int dim_size = this->getSize(dim);

    // check the shape_o
    DimVector shape_o = this->shape;
    // reset the tensor_o
    if(tensor_o == nullptr) {
        tensor_o = new Tensor(shape_o);
    } tensor_o->reset(shape_o);     // reset logits problem

    printShape(tensor_o->getShape());
    printShape(mean->getShape());

    this->sub(tensor_o, mean);
    tensor_o->square_();

    if(size == 2 && dim == 0) {
        size_t N = shape[0], C = shape[1];
        dim3 block(BLOCK_SIZE1D, BATCH_BASE);
        dim3 grid((C-1)/BLOCK_SIZE1D + 1);
        kBatchReduce2D<<<grid, block>>>(tensor_o->d_data, tensor_o->d_data, N, C, true);CHECK_KERNEL();
        shape_o[dim] = 1;
        tensor_o->reset(shape_o);
        if(!keepDim)
            tensor_o->squeeze(dim);
    } else if(size == 3 && dim == 0) {
        size_t N = shape[0], C = shape[1], L=shape[2];
        dim3 block(BLOCK_SIZE1D, BATCH_BASE);
        dim3 grid((C*L -1)/BLOCK_SIZE1D + 1);
        kBatchReduce2D<<<grid, block>>>(tensor_o->d_data, tensor_o->d_data, N, C*L, true);CHECK_KERNEL();
        shape_o[dim] = 1; 
        tensor_o->reset(shape_o);
        if(!keepDim)
            tensor_o->squeeze(dim);
    } else {
        ERROR("Not implemented!\n");
    }
    return tensor_o;
}

Tensor* Tensor::argmax(int dim, bool keepDim) {
    int size = this->shape.size();
    if(dim < 0) {
        dim = size + dim;
    }

    DimVector shape_o = this->shape;
    Tensor* tensor_o = new Tensor(shape_o);

    if(dim < size) {
        if(size == 2 && dim==size-1) {
            // used for getting the label
            size_t C=this->shape[0], L=this->shape[1];
            int block = BLOCK_SIZE1D;   // BLOCK_SIZE could be larger
            int grid = C;

            kArgmaxLastDim2D<<<grid, block>>>(this->d_data, tensor_o->d_data, C, L); CHECK_KERNEL();

            tensor_o->shape[dim] = 1;
            if(!keepDim)
                tensor_o->squeeze(-1);
                
            return tensor_o;
        } else {
            ERROR("Not implementated: size=%d dim=%d\n", size, dim);
        }
    } else {
        ERROR("size=%d<dim=%d\n", size, dim);
    }
    return tensor_o;
}

void Tensor::argmax(Tensor*& output, int dim, bool keepDim) {
    int size = this->shape.size();
    if(dim < 0) {
        dim = size + dim;
    }

    DimVector shape_o = this->shape;
    shape_o[dim] = 1;
    if(!keepDim) {
        shape_o.erase(std::remove(shape_o.begin(), shape_o.end(), 1), shape_o.end());
    }
    if (output == nullptr) {
        output = new Tensor(shape_o); 
    }
    output->reset(shape_o);

    if(dim < size) {
        if(size == 2 && dim==size-1) {
            // used for getting the label
            size_t C=this->shape[0], L=this->shape[1];
            int block = BLOCK_SIZE1D;   // BLOCK_SIZE could be larger
            int grid = C;

            kArgmaxLastDim2D<<<grid, block>>>(this->d_data, output->d_data, C, L); CHECK_KERNEL();
        } else {
            ERROR("Not implementated: size=%d dim=%d\n", size, dim);
        }
    } else {
        ERROR("size=%d<dim=%d\n", size, dim);
    }
}

void Tensor::max_(int dim, bool keepDim){
    int size = this->shape.size();
    if(dim < 0) {
        dim = size + dim;
    }

    if(dim < size) {
        // DEBUG_PRINT("size=%d, dim=%d\n", size, dim);
        if(size == 3 && dim==size-1) {
            size_t N=this->shape[0], C=this->shape[1], L=this->shape[2];
            int block = BLOCK_SIZE1D;   // BLOCK_SIZE could be larger
            dim3 grid(C, N);
            kMaxLastDim3D<<<grid, block>>>(this->d_data, this->d_data, N, C, L); CHECK_KERNEL();

            this->shape[dim] = 1;
            if(!keepDim)
                this->squeeze(-1);
        } else {
            ERROR("Not implementated!\n");
        }
    }
}


Tensor* Tensor::max(Tensor*& output, int dim, bool keepDim) {
    int size = this->shape.size();
    if (dim < 0) dim = size + dim;
    DimVector shape_o = this->shape;
    shape_o[dim] = 1;
    if(!keepDim) {
        shape_o.erase(std::remove(shape_o.begin(), shape_o.end(), 1), shape_o.end());
    }

    if (output == nullptr) {
        output = new Tensor(shape_o); 
    }
    output->reset(shape_o);


    if (dim < size) {
        if (size == 3 && dim == size - 1) {
            size_t N = this->shape[0], C = this->shape[1], L = this->shape[2];
            int block = BLOCK_SIZE1D;
            dim3 grid(C, N);
            kMaxLastDim3D<<<grid, block>>>(this->d_data, output->d_data, N, C, L); CHECK_KERNEL();

            output->shape[dim] = 1;
            if (!keepDim) {
                output->squeeze(-1);
            }
            return output;
        } else if (size == 2 && dim == size - 1) {
            size_t C = this->shape[0], L = this->shape[1];
            int block = BLOCK_SIZE1D;
            int grid = C;

            kMaxLastDim2D<<<grid, block>>>(this->d_data, output->d_data, C, L); CHECK_KERNEL();

            output->shape[dim] = 1;
            if (!keepDim) {
                output->squeeze(-1);
            }
            return output; // 直接返回传入的 output
        } else {
            ERROR("Not implemented!\n");
        }
    }

    return output; // 如果没有进入 kernel，直接返回 output
}

Tensor* Tensor::max_wt_index(Tensor*& output, Tensor*& max_index, int dim, bool keepDim) {
    int size = this->shape.size();
    if (dim < 0) dim = size + dim;
    DimVector shape_o = this->shape;
    shape_o[dim] = 1;
    if(!keepDim) {
        shape_o.erase(std::remove(shape_o.begin(), shape_o.end(), 1), shape_o.end());
    }

    if (output == nullptr) {
        output = new Tensor(shape_o); 
    } output->reset(shape_o);

    if(max_index == nullptr) {
        max_index = new Tensor(shape_o);
    }   max_index->reset(shape_o);

    if (dim < size) {
        if (size == 3 && dim == size - 1) {
            size_t N = this->shape[0], C = this->shape[1], L = this->shape[2];
            int block = BLOCK_SIZE1D;
            dim3 grid(C, N);
            kMaxIdxLastDim3D<<<grid, block>>>(this->d_data, output->d_data, max_index->getData(), N, C, L); CHECK_KERNEL();

            output->shape[dim] = 1;
            if (!keepDim) {
                output->squeeze(-1);
            }
            return output;
        } else if (size == 2 && dim == size - 1) {
            size_t C = this->shape[0], L = this->shape[1];
            int block = BLOCK_SIZE1D;
            int grid = C;

            kMaxIdxLastDim2D<<<grid, block>>>(this->d_data, output->d_data, max_index->getData(), C, L); CHECK_KERNEL();

            output->shape[dim] = 1;
            if (!keepDim) {
                output->squeeze(-1);
            }
            return output; // 直接返回传入的 output
        } else {
            ERROR("Not implemented!\n");
        }
    }

    return output; // 如果没有进入 kernel，直接返回 output
}


Tensor* Tensor::max(int dim, bool keepDim){
    int size = this->shape.size();
    if(dim < 0) {
        dim = size + dim;
    }

    DimVector shape_o = this->shape;
    Tensor* tensor_o = new Tensor(shape_o);

    if(dim < size) {
        if(size == 3 && dim==size-1) {
            size_t N=this->shape[0], C=this->shape[1], L=this->shape[2];
            int block = BLOCK_SIZE1D;   // BLOCK_SIZE could be larger
            dim3 grid(C, N);
            kMaxLastDim3D<<<grid, block>>>(this->d_data, tensor_o->d_data, N, C, L); CHECK_KERNEL();

            tensor_o->shape[dim] = 1;
            if(!keepDim)
                tensor_o->squeeze(-1);
            return tensor_o;
        } else if(size == 2 && dim==size-1) {
            size_t C=this->shape[0], L=this->shape[1];
            int block = BLOCK_SIZE1D;   // BLOCK_SIZE could be larger
            int grid = C;

            kMaxLastDim2D<<<grid, block>>>(this->d_data, tensor_o->d_data, C, L); CHECK_KERNEL();

            tensor_o->shape[dim] = 1;
            if(!keepDim)
                tensor_o->squeeze(-1);
            return tensor_o;
        } else {
            ERROR("Not implementated!\n");
        }
    }
    return tensor_o;
}


Tensor* Tensor::squeeze() {
    shape.erase(std::remove(shape.begin(), shape.end(), 1), shape.end());
    if(shape.size()==0) {
        this->shape = {1};
    }
    return this;
}

Tensor* Tensor::squeeze(int idx) {
    if(idx < 0) idx = shape.size() + idx;
    if(idx < shape.size() && shape[idx] == 1)
        shape.erase(shape.begin() + idx);
    if(shape.size()==0) {
        this->shape = {1};
    }
    return this;
}

Tensor* Tensor::unsqueeze(int idx) {
    if (idx >= 0 && idx <= shape.size()) {
        shape.insert(shape.begin() + idx, 1);
    } else if(idx < 0) {
        shape.insert(shape.end() + 1 + idx, 1);
    }
    return this;
}

/* Binary op */

void Tensor::saxpy_(Tensor* tensor, float f1, float f2) {
    size_t d1 = this->getDim(), d2 = tensor->getDim();
    // bool ge = d1 >= d2;
    // Tensor& mat1 = ge ? *this : *tensor;
    // Tensor& mat2 = ge ? *tensor : *this;
    // DimVector shape1 = ge ? this->shape : tensor->shape;
    // DimVector shape2 = ge ? tensor->shape : this->shape;
    DimVector shape1 = this->shape;
    DimVector shape2 = tensor->shape;
    size_t dim = d1 >= d2 ? d1 : d2;

    DimVector stride1(dim);
    DimVector stride2(dim);
    DimVector shape_o = getBroadcastShape(shape1, shape2, stride1, stride2);
    if(shape_o != this->getShape()) {
        printShape(shape1);
        printShape(shape2);
        ERROR("Inplace saxpy failed!");
    }

    if(shape_o.size() > 0) {
        if(dim == 1) {
            // e.x. shape1: (N) shape2: (N) or (1)
            int block = BLOCK_SIZE1D;
            int grid = (shape_o[0] - 1) / block + 1;
            int s1 = stride1[0], s2 = stride2[0];

            kAddStride_l1<<<grid, block>>>(this->d_data, tensor->d_data, this->d_data, this->getSize(), f1, f2, s1, s2); CHECK_KERNEL();
        } else if(dim == 2) {
            dim3 block(BLOCK_SIZE2D, BLOCK_SIZE2D);
            int row = shape_o[0], col = shape_o[1];
            dim3 grid((col-1)/BLOCK_SIZE2D+1, (row-1)/BLOCK_SIZE2D+1);

            // e.x. shape1: (B x N), shape2: (N) or else
            int s11 = stride1[0], s12 = stride1[1];
            int s21 = stride2[0], s22 = stride2[1];

            kAddStride_l2<<<grid, block>>>(this->d_data, tensor->d_data, this->d_data, row, col, f1 ,f2, s11, s12, s21, s22); CHECK_KERNEL();
        } else if(dim == 3) {
            dim3 block(BLOCK_SIZE3D, BLOCK_SIZE3D, BATCH_BASE);
            int z = shape_o[0], y = shape_o[1], x = shape_o[2];
            dim3 grid((x-1)/BLOCK_SIZE3D+1, (y-1)/BLOCK_SIZE3D+1, (z-1)/BATCH_BASE+1);

            // e.x. shape1: (B x N), shape2: (N) or else
            int s11 = stride1[0], s12 = stride1[1], s13 = stride1[2];
            int s21 = stride2[0], s22 = stride2[1], s23 = stride2[2];

            kAddStride_l3<<<grid, block>>>(this->d_data, tensor->d_data, this->d_data, z, y, x, f1 ,f2, s11, s12, s13, s21, s22, s23); CHECK_KERNEL();
        } else {
            ERROR("Not implemented!\n");
        }
    } else {
        ERROR("Failed to align the dimension!\n");
    }
}


void Tensor::add_(Tensor* tensor, float f1, float f2) {
    saxpy_(tensor, f1, f2);
}

void Tensor::sub_(Tensor* tensor, float f1, float f2) {
    saxpy_(tensor, f1, -f2);
}

Tensor* Tensor::saxpy(Tensor*& tensor_o, Tensor* tensor, float f1, float f2) {
    size_t d1 = this->getDim(), d2 = tensor->getDim();
    DimVector shape1 = this->shape;
    DimVector shape2 = tensor->shape;
    size_t dim = d1 >= d2 ? d1 : d2;

    DimVector stride1(dim);
    DimVector stride2(dim);
    DimVector shape_o = getBroadcastShape(shape1, shape2, stride1, stride2);

    if(tensor_o == nullptr) {
        tensor_o = new Tensor(shape_o);
    } tensor_o->reset(shape_o);

    if(shape_o.size() > 0) {
        if(dim == 1) {
            // e.x. shape1: (N) shape2: (N) or (1)
            int block = BLOCK_SIZE1D;
            int grid = (shape_o[0] - 1) / block + 1;
            int s1 = stride1[0], s2 = stride2[0];

            kAddStride_l1<<<grid, block>>>(this->d_data, tensor->d_data, tensor_o->d_data, this->getSize(), f1, f2, s1, s2); CHECK_KERNEL();
        } else if(dim == 2) {
            dim3 block(BLOCK_SIZE1D, BATCH_BASE);
            int row = shape_o[0], col = shape_o[1];
            dim3 grid((col-1)/BLOCK_SIZE1D+1, (row-1)/BATCH_BASE+1);

            // e.x. shape1: (B x N), shape2: (N) or else
            int s11 = stride1[0], s12 = stride1[1];
            int s21 = stride2[0], s22 = stride2[1];

            kAddStride_l2<<<grid, block>>>(this->d_data, tensor->d_data, tensor_o->d_data, row, col, f1 ,f2, s11, s12, s21, s22); CHECK_KERNEL();
        } else if(dim == 3) {
            dim3 block(BLOCK_SIZE2D, BLOCK_SIZE2D, BATCH_BASE);
            int z = shape_o[0], y = shape_o[1], x = shape_o[2];
            dim3 grid((x-1)/BLOCK_SIZE2D+1, (y-1)/BLOCK_SIZE2D+1, (z-1)/BATCH_BASE+1);

            // e.x. shape1: (B x N), shape2: (N) or else
            int s11 = stride1[0], s12 = stride1[1], s13 = stride1[2];
            int s21 = stride2[0], s22 = stride2[1], s23 = stride2[2];

            kAddStride_l3<<<grid, block>>>(this->d_data, tensor->d_data, tensor_o->d_data, z, y, x, f1 ,f2, s11, s12, s13, s21, s22, s23); CHECK_KERNEL();
        } else {
            ERROR("Not implemented!\n");
        }
    } else {
        ERROR("Failed to align the dimension!\n");
    }
    return tensor_o;  
}

Tensor* Tensor::add(Tensor*& output, Tensor* tensor) {
    return saxpy(output, tensor, 1.0f, 1.0f);
}

Tensor* Tensor::sub(Tensor*& output, Tensor* tensor) {
    return saxpy(output, tensor, 1.0f, -1.0f);
}

Tensor* Tensor::saxpy_plus(Tensor*& tensor_o, Tensor* tensor, float factor, int flag) {
    size_t d1 = this->getDim(), d2 = tensor->getDim();
    // bool ge = d1 >= d2;
    // Tensor& mat1 = ge ? *this : *tensor;
    // Tensor& mat2 = ge ? *tensor : *this;
    // DimVector shape1 = ge ? this->shape : tensor->shape;
    // DimVector shape2 = ge ? tensor->shape : this->shape;
    DimVector shape1 = this->shape;
    DimVector shape2 = tensor->shape;
    size_t dim = d1 >= d2 ? d1 : d2;

    DimVector stride1(dim);
    DimVector stride2(dim);
    DimVector shape_o = getBroadcastShape(shape1, shape2, stride1, stride2);
    // Tensor* tensor_o = new Tensor(shape_o);
    if(tensor_o == nullptr) {
        tensor_o = new Tensor(shape_o);
    } tensor_o->reset(shape_o);

    if(shape_o.size() > 0) {
        if(dim == 1) {
            // e.x. shape1: (N) shape2: (N) or (1)
            int block = BLOCK_SIZE1D;
            int grid = (shape_o[0] - 1) / block + 1;
            int s1 = stride1[0], s2 = stride2[0];
            if(flag) {
                kDotStride_l1<<<grid, block>>>(this->d_data, tensor->d_data, tensor_o->d_data, this->getSize(), factor, s1, s2); CHECK_KERNEL();
            } else {
                kDivStride_l1<<<grid, block>>>(this->d_data, tensor->d_data, tensor_o->d_data, this->getSize(), factor, s1, s2); CHECK_KERNEL();
            }
        } else if(dim == 2) {
            dim3 block(BLOCK_SIZE2D, BLOCK_SIZE2D);
            int row = shape_o[0], col = shape_o[1];
            dim3 grid((col-1)/BLOCK_SIZE2D+1, (row-1)/BLOCK_SIZE2D+1);

            // e.x. shape1: (B x N), shape2: (N) or else
            int s11 = stride1[0], s12 = stride1[1];
            int s21 = stride2[0], s22 = stride2[1];
            if(flag) {
                kDotStride_l2<<<grid, block>>>(this->d_data, tensor->d_data, tensor_o->d_data, row, col, factor, s11, s12, s21, s22); CHECK_KERNEL();
            } else {
                kDivStride_l2<<<grid, block>>>(this->d_data, tensor->d_data, tensor_o->d_data, row, col, factor, s11, s12, s21, s22); CHECK_KERNEL();
            }
        } else if(dim == 3) {
            dim3 block(BLOCK_SIZE3D, BLOCK_SIZE3D, BATCH_BASE);
            int z = shape_o[0], y = shape_o[1], x = shape_o[2];
            dim3 grid((x-1)/BLOCK_SIZE3D+1, (y-1)/BLOCK_SIZE3D+1, (z-1)/BATCH_BASE+1);

            // e.x. shape1: (B x N), shape2: (N) or else
            int s11 = stride1[0], s12 = stride1[1], s13 = stride1[2];
            int s21 = stride2[0], s22 = stride2[1], s23 = stride2[2];
            if(flag) {
                kDotStride_l3<<<grid, block>>>(this->d_data, tensor->d_data, tensor_o->d_data, z, y, x, factor, s11, s12, s13, s21, s22, s23); CHECK_KERNEL();
            } else {
                kDivStride_l3<<<grid, block>>>(this->d_data, tensor->d_data, tensor_o->d_data, z, y, x, factor, s11, s12, s13, s21, s22, s23); CHECK_KERNEL();
            }
        } else {
            ERROR("Not implemented!\n");
        }
    } else {
        printShape(this->shape);
        printShape(tensor->shape);
        ERROR("Failed to align the dimension!\n");
    }
    return tensor_o;  
}

Tensor* Tensor::dot(Tensor*& output, Tensor* tensor, float factor) {
    return saxpy_plus(output, tensor, factor, 1);
}

Tensor* Tensor::div(Tensor*& output, Tensor* tensor, float factor) {
    return saxpy_plus(output, tensor, factor, 0);
}

Tensor* Tensor::matmul(Tensor*& tensor_o, Tensor* tensor) {
    size_t d1 = this->getDim(), d2 = tensor->getDim();
    bool ge = d1 >= d2;
    Tensor& mat1 = ge ? *this : *tensor;
    Tensor& mat2 = ge ? *tensor : *this;
    DimVector shape1 = ge ? this->shape : tensor->shape;
    DimVector shape2 = ge ? tensor->shape : this->shape;
    size_t dim1 = shape1.size(), dim2 = shape2.size();

    DimVector shape_o = getMatmulShape(shape1, shape2);

    if(!shape_o.empty()) {
        if(tensor_o == nullptr) {
            tensor_o = new Tensor(shape_o);
        } tensor_o->reset(shape_o);

        if(dim1 == 1) {
            assert(shape1[0] == shape2[0]);
            int block = BLOCK_SIZE1D;
            int grid = (shape1[0] - 1)/block + 1;

            // the requirement of implementation
            kMatmul_l1<<<grid, block>>>((&mat1)->d_data, (&mat2)->d_data, tensor_o->getData(), this->getSize()); CHECK_KERNEL();
        } else if(dim1 == 2) {
            if(dim2 == 1) {
                int block = BLOCK_SIZE1D;
                dim3 grid(1, (shape1[0] - 1) / block + 1);
                int M = shape1[0], N = shape1[1];
                kMatmul_l2<<<grid, block>>>((&mat1)->d_data, (&mat2)->d_data, tensor_o->d_data, M, N); CHECK_KERNEL();
            } else if(dim2 == 2) {
                dim3 block(BLOCK_SIZE2D, BLOCK_SIZE2D);
                int x = shape_o[0], y = shape_o[1];
                dim3 grid((y-1)/BLOCK_SIZE2D + 1, (x-1)/BLOCK_SIZE2D + 1);
                int M = shape1[0], N = shape1[1], K = shape2[1];
                // kMatmulTransposed_l3<<<grid, block>>>((&mat1)->d_data, (&mat2)->d_data, tensor_o->d_data, M, N, K);CHECK_KERNEL();
                kMatmul_l3<<<grid, block>>>((&mat1)->d_data, (&mat2)->d_data, tensor_o->d_data, M, N, K);CHECK_KERNEL();
            } else {
                ERROR("Not implemented!");
            }
        } else {
            ERROR("Not implemented!");
        }
        return tensor_o;

    } else {
        printShape(this->shape);
        printShape(tensor->shape);
        ERROR("Failed to match mats' shape!");
    }
    return tensor_o;
}

/* (B x M x N) @ (B x N x K) = (B x M x K) */
Tensor* Tensor::bmm(Tensor* tensor) {
    size_t d1 = this->getDim(), d2 = tensor->getDim();
    bool ge = d1 >= d2;
    size_t dim = ge ? d1 : d2;

    // DimVector shape_o = getBatchMatmulShape(shape1, shape2);
    size_t bz = this->shape[0];
    if(dim != 3 || tensor->shape[0] != bz || this->shape[2] != tensor->shape[1]) {
        ERROR("bmm shape not matched!\n");
    }

    size_t M = this->shape[1], N = this->shape[2], K = tensor->shape[2];
    Tensor* tensor_o = new Tensor({bz, M, K});

    dim3 block(BLOCK_SIZE2D, BLOCK_SIZE2D);
    dim3 grid((K-1)/BLOCK_SIZE2D +1, (M-1)/BLOCK_SIZE2D+1, bz);

    kBatchMatmul3D<<<grid, block>>>(this->d_data, tensor->d_data, tensor_o->d_data, bz, M, N, K); CHECK_KERNEL();

    return tensor_o;
}

/* (B x M x N) @ (B x N x K) = (B x M x K) */
void Tensor::bmm(Tensor*& output, Tensor* tensor) {
    size_t d1 = this->getDim(), d2 = tensor->getDim();
    bool ge = d1 >= d2;
    size_t dim = ge ? d1 : d2;

    size_t bz = this->shape[0];
    size_t M = this->shape[1], N = this->shape[2], K = tensor->shape[2];
    DimVector shape_o = {bz, M, K};

    if (output == nullptr) {
        output = new Tensor(shape_o); 
    }
    output->reset(shape_o);

    if(dim != 3 || tensor->shape[0] != bz || this->shape[2] != tensor->shape[1])  {
        printShape(this->shape);
        printShape(tensor->shape);
        ERROR("bmm shape not matched!\n");
    }

    dim3 block(BLOCK_SIZE2D, BLOCK_SIZE2D);
    dim3 grid((K-1)/BLOCK_SIZE2D +1, (M-1)/BLOCK_SIZE2D+1, bz);

    kBatchMatmul3D<<<grid, block>>>(this->d_data, tensor->d_data, output->d_data, bz, M, N, K); CHECK_KERNEL();
}

void Tensor::oneHot(int* d_labels, Tensor*& target, size_t label_num, size_t class_num, cudaStream_t stream) {
    DimVector shape_o = {label_num, class_num};

    if(target == nullptr) {
        target = new Tensor(shape_o);
    }
    target->reset(shape_o);

    int block = BLOCK_SIZE1D;
    dim3 grid((class_num - 1)/BLOCK_SIZE1D + 1, label_num);
    kOneHot<<<grid, block, 0, stream>>>(d_labels, target->getData(), label_num, class_num); CHECK_KERNEL();
}


/* tool funcs */

bool isContinuousSqueeze(DimVector& shape1, DimVector& shape2) {
    size_t dim1 = shape1.size();
    size_t dim2 = shape2.size();

    int i = dim1 - 1, j = dim2 - 1;
    int acc_1 = 0, acc_2 = 0;

    while(i >= 0 && j >= 0) {
        // DEBUG_PRINT("shape1[%d] = %ld, shape2[%d] = %ld, acc1=%d, acc2=%d\n", i, shape1[i], j, shape2[j], acc_1, acc_2);
        if(shape1[i]==1) {
            i--; continue;
        } else if(shape2[j] == 1) {
            j--; continue;
        }
        if(acc_1 == 0 && acc_2 == 0) {
            if (shape1[i] == shape2[j]){
                i--; j--;
            } else {
                if (shape1[i] > shape2[j]) {
                    acc_2 = shape2[j];
                    j--;
                } else {
                    acc_1 = shape1[i];
                    i--;
                }
            }
        } else {
            if (acc_1 != 0) {
                acc_1 *= shape1[i];
                if (acc_1 == shape2[j]) {
                    i--; j--;
                    acc_1 = 0;
                } else if(acc_1 < shape2[j]) {
                    i--;
                } else {
                    ERROR("Shape squeeze invalid!\n");
                }
            } else if(acc_2 != 0) {
                acc_2 *= shape2[j];
                if (acc_2 == shape1[i]) {
                    i--; j--;
                    acc_2 = 0;
                } else if(acc_2 < shape1[i]) {
                    j--;
                } else {
                    DEBUG_PRINT("acc_2 = %d, shape[%d]=%ld\n", acc_2, i, shape1[i]);
                    ERROR("Shape squeeze invalid!\n");
                }
            } else {
                ERROR("Shape squeeze invalid!\n");
            }
        }
    }

    while(i>=0) {
        if (shape1[i] == 1) i--;
    }

    while(j>=0) {
        if (shape2[j] == 1) j--;
    }

    return i < 0 && j < 0; 
}

/* only consider supplementing matrix b into the shape of matrix a */
DimVector getBroadcastShape(DimVector& shape1, DimVector& shape2, DimVector& stride1, DimVector& stride2) {
    int dim1 = shape1.size(), dim2 = shape2.size();
    // assert(dim1 >= dim2);   
    int i = dim1-1;
    int j = dim2-1;
    int align_idx = dim1 >= dim2 ? dim1 : dim2;
    DimVector shape_o(align_idx);
    align_idx--;

    while(align_idx >= 0) {
        if(i < 0) {
            stride1[align_idx] = shape2[j];
            stride2[align_idx] = 1;
            shape_o[align_idx] = shape2[j];
            j--;
        } else if(j < 0) {
            stride1[align_idx] = 1;
            stride2[align_idx] = shape1[i];
            shape_o[align_idx] = shape1[i];
            i--;
        } else if(shape1[i] == shape2[j]){
            stride1[align_idx] = 1;
            stride2[align_idx] = 1;
            shape_o[align_idx] = shape1[i];
            i--; j--;
        }
        else if(shape1[i] == 1) {
            stride1[align_idx] = shape2[j];
            stride2[align_idx] = 1; // valid
            shape_o[align_idx] = shape2[j];
            i--; j--;
        } else if(shape2[j] == 1) {
            stride1[align_idx] = 1; // valid
            stride2[align_idx] = shape1[i]; 
            shape_o[align_idx] = shape1[i];
            i--; j--;
        } else if(shape1[i] != shape2[j]) {
            return {};
        }
        align_idx--;
    }

    return shape_o;
}

DimVector getMatmulShape(DimVector& shape1, DimVector& shape2) {
    DimVector vec = shape2;
    int dim1 = shape1.size(), dim2 = shape2.size();
    int dim = dim1 > dim2 ? dim1 : dim2;
    if(dim == 1) {
        if(shape1[0] != shape2[0]) 
            return {1};
    } else if(dim == 2) {
        if(dim1 == dim2) {
            std::reverse(vec.begin(), vec.end());
            if (shape1[1] == vec[1]) {
                return {shape1[0], vec[0]};
            }
        } else if(dim1 == 2) {
            // to be optimize: shape1[0] == 0
            if(shape1[1] == shape2[0]) {
                return {shape1[0]};
            }
        } else if(dim2 == 2) {
            if(shape1[1] == shape2[0]) {
                return {shape2[0]};
            }
        }
    } else {
        printShape(shape1);
        printShape(shape2);
        ERROR("Not implemented!");
    }
    return {};
}

DimVector getBatchMatmulShape(DimVector& shape1, DimVector& shape2) {
    int dim1 = shape1.size(), dim2 = shape2.size();
    int dim = dim1 > dim2 ? dim1 : dim2;
    if(dim == 1) {
        if(shape1[0] != shape2[0]) 
            return {1};
    } else if(dim == 2) {
        if(dim1 == dim2) {
            if (shape1[1] == shape2[1]) {
                return {shape1[0], shape2[0]};
            }
        } else if(dim1 == 2) {
            // to be optimize: shape1[0] == 0
            if(shape1[1] == shape2[0]) {
                return {shape1[0]};
            }
        } else if(dim2 == 2) {
            if(shape1[1] == shape2[0]) {
                return {shape2[0]};
            }
        }
    } else {
        ERROR("Not implemented!");
    }
    return {};
}

