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
    size_t ndata = 1;
    for(int i=0; i<dim; i++) {
        ndata *= shape[i];
    }
    this->n_data = ndata;
    CHECK(cudaMalloc(&(this->d_data), ndata * sizeof(float)));

    this->initialize(init_type);
}

Tensor::Tensor(float *h_data, DimVector shape) {
    int dim = shape.size();
    this->shape = shape;
    size_t ndata = 1;
    for(int i=0; i<dim; i++) {
        ndata *= shape[i];
    }
    this->n_data = ndata;
    size_t nBytes = ndata * sizeof(float);
    CHECK(cudaMalloc((float **)&(this->d_data), nBytes));
    CHECK(cudaMemcpy(d_data, h_data, nBytes, cudaMemcpyHostToDevice));
}

Tensor::~Tensor(){
    CHECK(cudaFree(this->d_data));
}

void Tensor::fromVec(std::vector<float>& vec) {
    if(vec.size() != this->n_data) {
        ERROR("%ld != %ld, weight size not matched!\n", vec.size(), this->n_data);
    }
    float* h_data = vec.data();
    CHECK(cudaMemcpy(this->d_data, h_data, this->n_data * sizeof(float), cudaMemcpyHostToDevice));
}

std::vector<float> Tensor::toVec() {
    float* h_data = this->toHost();
    size_t size = this->getDataNum();
    std::vector<float> vec(h_data, h_data + size);
    return vec;
}

/* getters */

size_t Tensor::getDim() {
    return this->shape.size();
}

size_t Tensor::getDataNum() {
    return this->n_data;
}

DimVector Tensor::getShape(){
    return this->shape;
}

size_t Tensor::getSize(size_t dim) {
    return this->shape.at(dim);
}

float* Tensor::toHost(){
    size_t nBytes = this->n_data * sizeof(float);
    float* h_d = (float *) malloc(nBytes);
    CHECK(cudaMemcpy(h_d, d_data, nBytes, cudaMemcpyDeviceToHost));
    return h_d;
}

float* Tensor::getData() {
    return this->d_data;
}

/* setters */

void Tensor::load(float* h_data, size_t n_data) {
    assert(n_data = this->n_data);
    CHECK(cudaMemcpy(d_data, h_data, n_data * sizeof(float), cudaMemcpyHostToDevice));
}

void Tensor::initialize(float value) {
    size_t nBytes = this->n_data * sizeof(float);
    float* h_d = (float *)malloc(nBytes);
    for (int i=0; i<n_data; i++) {
        h_d[i] = value;
    }
    CHECK(cudaMemcpy(d_data, h_d, nBytes, cudaMemcpyHostToDevice));
}

void Tensor::initialize(InitType type) {
    size_t nBytes = this->n_data * sizeof(float);
    float* h_d = (float *)malloc(nBytes);
    
    if(type == ZERO) {
        for (int i=0; i<n_data; i++) {
            h_d[i] = 0.0f;
        }
    } else if(type==ONES) {
        for (int i=0; i<n_data; i++) {
            h_d[i] = 1.0f;
        }
    } else if(type==IDENTITY) {
        assert(shape.size() == 2 && shape[0] == shape[1]);
        int s = shape[0];
        for (int i=0; i < s; i++) {
            for(int j=0; j < s; j++) {
                if(i==j) h_d[i*s + j] = 1.0f;
                else     h_d[i*s + j] = 0.0f;
            }
        }
    } else if(type==RANDOM) {
        float MAX = 1.0f;
        float MIN = -1.0f;
        for (int i=0; i<n_data; i++) {
            h_d[i] = randomFloat(MIN, MAX);
        }
    } else if(type==KAIMING) {
        assert(this->getDim() == 2);
        size_t in_features = this->shape[1]; // out_feature, in_feature
        float sqrt_k = 1.0f/(sqrt(in_features));
        float MAX = sqrt_k;
        float MIN = -sqrt_k;
        for (int i=0; i<n_data; i++) {
            h_d[i] = randomFloat(MIN, MAX);
        }
    }else {
        ERROR("Not implemented!");
    }
    
    CHECK(cudaMemcpy(d_data, h_d, nBytes, cudaMemcpyHostToDevice));
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

    if(dim_n == dim) {
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
    this->shape = {this->n_data};
}

void Tensor::transpose() {
    if(this->shape.size() == 2) {
        DimVector shape_o = this->shape;
        std::swap(shape_o[1], shape_o[2]);
        Tensor* tensor_o = new Tensor(shape_o);
        
        int row = this->shape[0], col = this->shape[1];
        dim3 block(BLOCK_SIZE2D, BLOCK_SIZE2D);
        // ATTENTION!!!
        dim3 grid((col-1)/BLOCK_SIZE2D + 1, (row-1)/BLOCK_SIZE2D + 1);

        kTranspose<<<grid, block>>>(this->d_data, tensor_o->d_data, row, col);
        CHECK_KERNEL();

        cudaFree(this->d_data);
        this->d_data = tensor_o->d_data;
        this->shape = shape_o;
    } else {
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
        DEBUG_PRINT("HERE, dim1=%d, dim2=%d\n", dim1, dim2);
        this->transpose();
    } else if(dim == 3) {
        if(dim2 == dim - 1) {
            DimVector shape_o = this->shape;
            std::swap(shape_o[1], shape_o[2]);
            Tensor* tensor_o = new Tensor(shape_o);

            size_t N = shape[0], row = shape[1], col = shape[2];
            dim3 block(BLOCK_SIZE2D, BLOCK_SIZE2D);
            dim3 grid((col-1)/BLOCK_SIZE2D + 1, (row-1)/BLOCK_SIZE2D+1);

            kTransposeLast3D<<<grid, block>>>(this->d_data, tensor_o->d_data, N, row, col); CHECK_KERNEL();

            cudaFree(this->d_data);
            this->d_data = tensor_o->d_data;
            this->shape = shape_o;
        } else {
            ERROR("dim1=%d, dim2=%d, Not implemented!\n", dim1, dim2);
        }
    }
}

void Tensor::scale_(float factor) {
    dim3 block(BLOCK_SIZE1D);
    dim3 grid((this->n_data-1)/BLOCK_SIZE1D + 1);
    // need padding?
    kScale<<<grid, block>>>(this->d_data, factor, 0.0f, this->n_data);
    CHECK_KERNEL();
}

Tensor* Tensor::exp() {
    DimVector shape_o = this->shape;
    Tensor* tensor_o = new Tensor(shape_o);
    int n_data = this->n_data;
    int block = BLOCK_SIZE1D;
    int grid = (n_data - 1)/BLOCK_SIZE1D + 1;

    kExp<<<grid, block>>>(this->d_data, tensor_o->d_data, n_data); CHECK_KERNEL();
    return tensor_o;
}

void Tensor::exp_() {
    int n_data = this->n_data;
    int block = BLOCK_SIZE1D;
    int grid = (n_data - 1)/BLOCK_SIZE1D + 1;

    kExp<<<grid, block>>>(this->d_data, this->d_data, n_data); CHECK_KERNEL();
}

Tensor* Tensor::log() {
    DimVector shape_o = this->shape;
    Tensor* tensor_o = new Tensor(shape_o);
    int n_data = this->n_data;
    int block = BLOCK_SIZE1D;
    int grid = (n_data - 1)/BLOCK_SIZE1D + 1;

    kLog<<<grid, block>>>(this->d_data, tensor_o->d_data, n_data); CHECK_KERNEL();
    return tensor_o;
}

void Tensor::log_() {
    int n_data = this->n_data;
    int block = BLOCK_SIZE1D;
    int grid = (n_data - 1)/BLOCK_SIZE1D + 1;

    kLog<<<grid, block>>>(this->d_data, this->d_data, n_data); CHECK_KERNEL();
}

void Tensor::add_(float c) {
    dim3 block(BLOCK_SIZE1D);
    dim3 grid((this->n_data-1)/BLOCK_SIZE1D + 1);
    // need padding?
    kScale<<<grid, block>>>(this->d_data, 1.0f, c, this->n_data);
    CHECK_KERNEL();
}

void Tensor::sub_(float c) {
    dim3 block(BLOCK_SIZE1D);
    dim3 grid((this->n_data-1)/BLOCK_SIZE1D + 1);
    // need padding?
    kScale<<<grid, block>>>(this->d_data, 1.0f, -c, this->n_data);
    CHECK_KERNEL();
}

float Tensor::sum(){
    int block_num = (this->n_data-1)/BLOCK_SIZE1D + 1;
    float* d_out;
    CHECK(cudaMalloc((float**)&d_out, block_num * sizeof(float)));

    kSum<<<block_num, BLOCK_SIZE1D>>>(this->d_data, d_out, this->n_data);

    size_t nBytes = sizeof(float);
    float* h_out = (float *)malloc(nBytes);
    CHECK(cudaMemcpy(h_out, d_out, nBytes, cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_out));
    return *h_out;
}

Tensor* Tensor::sum(int dim){
    int size = this->shape.size();
    if(dim < 0) {
        dim = dim + this->getDim();
    }
    if(size == 2 && dim == this->getDim() - 1) {
        DimVector shape_o = this->shape;
        Tensor* tensor_o = new Tensor(shape_o);
        size_t C = shape[0], L = shape[1];
        int block = BLOCK_SIZE1D;
        int grid = (C-1)/BLOCK_SIZE1D + 1;
        kSumLastDim2D<<<grid, block>>>(this->d_data, tensor_o->d_data, C, L);CHECK_KERNEL();
        return tensor_o;
    } else {
        ERROR("Not implemented!\n");
    }
    return nullptr;
}
 
float Tensor::mean(){
    float s = sum();
    return s/this->n_data;
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

            this->n_data /= this->shape[dim];
            this->shape[dim] = 1;
            if(!keepDim) {
                this->shape.erase(shape.begin() + dim);
            }
        } else {
            ERROR("Not implementated!\n");
        }
    }
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

            tensor_o->n_data /= shape[dim];
            tensor_o->shape[dim] = 1;
            if(!keepDim) {
                tensor_o->shape.erase(shape.begin() + dim);
            }
            return tensor_o;
        } else if(size == 2 && dim==size-1) {
            size_t C=this->shape[0], L=this->shape[1];
            int block = BLOCK_SIZE1D;   // BLOCK_SIZE could be larger
            int grid = (C-1)/BLOCK_SIZE1D + 1;

            kMaxLastDim2D<<<grid, block>>>(this->d_data, tensor_o->d_data, C, L); CHECK_KERNEL();

            tensor_o->n_data /= shape[dim];
            tensor_o->shape[dim] = 1;
            if(!keepDim) {
                tensor_o->shape.erase(shape.begin() + dim);
            }
            return tensor_o;
        } else {
            ERROR("Not implementated!\n");
        }
    }
    return tensor_o;
}


void Tensor::squeeze() {
    shape.erase(std::remove(shape.begin(), shape.end(), 1), shape.end());
}

void Tensor::squeeze(int idx) {
    if(idx < shape.size() && shape[idx] == 1)
        shape.erase(shape.begin() + idx);
}

void Tensor::unsqueeze(int idx) {
    if (idx >= 0 && idx <= shape.size()) {
        shape.insert(shape.begin() + idx, 1);
    } else if(idx < 0) {
        shape.insert(shape.end() + 1 + idx, 1);
    }
}

/* Binary op */

Tensor* Tensor::saxpy(Tensor* tensor, float f1, float f2) {
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
    Tensor* tensor_o = new Tensor(shape_o);

    if(shape_o.size() > 0) {
        if(dim == 1) {
            // e.x. shape1: (N) shape2: (N) or (1)
            int block = BLOCK_SIZE1D;
            int grid = (shape_o[0] - 1) / block + 1;
            int s1 = stride1[0], s2 = stride2[0];

            kAddStride_l1<<<grid, block>>>(this->d_data, tensor->d_data, tensor_o->d_data, this->n_data, f1, f2, s1, s2); CHECK_KERNEL();
        } else if(dim == 2) {
            dim3 block(BLOCK_SIZE2D, BLOCK_SIZE2D);
            int row = shape_o[0], col = shape_o[1];
            dim3 grid((col-1)/BLOCK_SIZE2D+1, (row-1)/BLOCK_SIZE2D+1);

            // e.x. shape1: (B x N), shape2: (N) or else
            int s11 = stride1[0], s12 = stride1[1];
            int s21 = stride2[0], s22 = stride2[1];

            kAddStride_l2<<<grid, block>>>(this->d_data, tensor->d_data, tensor_o->d_data, row, col, f1 ,f2, s11, s12, s21, s22); CHECK_KERNEL();
        } else if(dim == 3) {
            dim3 block(BLOCK_SIZE3D, BLOCK_SIZE3D, BLOCK_SIZE3D);
            int z = shape_o[0], y = shape_o[1], x = shape_o[2];
            dim3 grid((x-1)/BLOCK_SIZE3D+1, (y-1)/BLOCK_SIZE3D+1, (z-1)/BLOCK_SIZE3D+1);

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

Tensor* Tensor::add(Tensor* tensor) {
    return saxpy(tensor, 1.0f, 1.0f);
}

Tensor* Tensor::sub(Tensor* tensor) {
    return saxpy(tensor, 1.0f, -1.0f);
}

Tensor* Tensor::saxpy_plus(Tensor* tensor, float factor, int flag) {
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
    Tensor* tensor_o = new Tensor(shape_o);

    if(shape_o.size() > 0) {
        if(dim == 1) {
            // e.x. shape1: (N) shape2: (N) or (1)
            int block = BLOCK_SIZE1D;
            int grid = (shape_o[0] - 1) / block + 1;
            int s1 = stride1[0], s2 = stride2[0];
            if(flag) {
                kDotStride_l1<<<grid, block>>>(this->d_data, tensor->d_data, tensor_o->d_data, this->n_data, factor, s1, s2); CHECK_KERNEL();
            } else {
                kDivStride_l1<<<grid, block>>>(this->d_data, tensor->d_data, tensor_o->d_data, this->n_data, factor, s1, s2); CHECK_KERNEL();
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
            dim3 block(BLOCK_SIZE3D, BLOCK_SIZE3D, BLOCK_SIZE3D);
            int z = shape_o[0], y = shape_o[1], x = shape_o[2];
            dim3 grid((x-1)/BLOCK_SIZE3D+1, (y-1)/BLOCK_SIZE3D+1, (z-1)/BLOCK_SIZE3D+1);

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
        ERROR("Failed to align the dimension!\n");
    }
    return tensor_o;  
}

Tensor* Tensor::dot(Tensor* tensor, float factor) {
    return saxpy_plus(tensor, factor, 1);
}

Tensor* Tensor::div(Tensor* tensor, float factor) {
    return saxpy_plus(tensor, factor, 0);
}

Tensor* Tensor::matmul(Tensor* tensor) {
    size_t d1 = this->getDim(), d2 = tensor->getDim();
    bool ge = d1 >= d2;
    Tensor& mat1 = ge ? *this : *tensor;
    Tensor& mat2 = ge ? *tensor : *this;
    DimVector shape1 = ge ? this->shape : tensor->shape;
    DimVector shape2 = ge ? tensor->shape : this->shape;
    size_t dim1 = shape1.size(), dim2 = shape2.size();

    DimVector shape_o = getMatmulShape(shape1, shape2);

    if(!shape_o.empty()) {
        Tensor* tensor_o = new Tensor(shape_o);
        tensor_o->initialize(0);
        if(dim1 == 1) {
            assert(shape1[0] == shape2[0]);
            int block = BLOCK_SIZE1D;
            int grid = (shape1[0] - 1)/block + 1;

            // the requirement of implementation
            float* d_tmp;
            CHECK(cudaMalloc((float **)&d_tmp, grid * sizeof(float)));
            kMatmul_l1<<<grid, block>>>((&mat1)->d_data, (&mat2)->d_data, d_tmp, this->n_data);
            CHECK_KERNEL();
            CHECK(cudaMemcpy(tensor_o->d_data, d_tmp, sizeof(float), cudaMemcpyDeviceToDevice));
            CHECK(cudaFree(d_tmp));
        } else if(dim1 == 2) {
            if(dim2 == 1) {
                int block = BLOCK_SIZE1D;
                dim3 grid(1, (shape1[0] - 1) / block + 1);
                int M = shape1[0], N = shape1[1];
                kMatmul_l2<<<grid, block>>>((&mat1)->d_data, (&mat2)->d_data, tensor_o->d_data, M, N);
                CHECK_KERNEL();
            } else {
                dim3 block(BLOCK_SIZE2D, BLOCK_SIZE2D);
                int x = shape_o[0], y = shape_o[1];
                dim3 grid((y-1)/BLOCK_SIZE2D + 1, (x-1)/BLOCK_SIZE2D + 1);
                int M = shape1[0], N = shape1[1], K = shape2[0];
                kMatmulTransposed_l3<<<grid, block>>>((&mat1)->d_data, (&mat2)->d_data, tensor_o->d_data, M, N, K);
                CHECK_KERNEL();
            }
        } else {
            ERROR("Not implemented!");
        }
        return tensor_o;

    } else {
        ERROR("Failed to match mats' shape!");
    }
    return nullptr;
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
    dim3 grid((K-1)/BLOCK_SIZE2D +1, (M-1)/BLOCK_SIZE2D+1);

    kBatchMatmul3D<<<grid, block>>>(this->d_data, tensor->d_data, tensor_o->d_data, bz, M, N, K); CHECK_KERNEL();

    return tensor_o;
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

