#include "tensor.cuh"

// Declarations
bool isContinuousSqueeze(DimVector& shape1, DimVector& shape2);
bool isBroadcast(DimVector& shape1, DimVector& shape2);
DimVector getBroadcastShape(DimVector& shape1, DimVector& shape2);
bool checkMatmulShape(DimVector& shape1, DimVector& shape2);
DimVector getMatmulShape(DimVector& shape1, DimVector& shape2);
DimVector getBatchMatmulShape(DimVector& shape1, DimVector& shape2);

Tensor::Tensor(DimVector& shape, InitType init_type) {
    this->shape = shape;
    int dim = shape.size();
    size_t ndata = 1;
    for(int i=0; i<dim; i++) {
        ndata *= shape[i];
        this->shape[i] = shape[i];
    }
    this->n_data = ndata;
    CHECK(cudaMalloc(&(this->d_data), ndata * sizeof(float)));

    this->initialize(init_type);
}

Tensor::Tensor(float *h_data, DimVector& shape) {
    int dim = shape.size();
    this->shape = DimVector(dim);
    size_t ndata = 1;
    for(int i=0; i<dim; i++) {
        ndata *= shape[i];
        this->shape[i] = shape[i];
    }
    this->n_data = ndata;
    size_t nBytes = ndata * sizeof(float);
    CHECK(cudaMalloc((float **)&(this->d_data), nBytes));
    CHECK(cudaMemcpy(d_data, h_data, nBytes, cudaMemcpyHostToDevice));
}

Tensor::~Tensor(){
    CHECK(cudaFree(this->d_data));
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
            for(int j=0; j<s; j++) {
                if(i==j) h_d[i*s + j] = 1.0f;
                else     h_d[i*s + j] = 0.0f;
            }
        }
    } else if(type==RANDOM) {
        float MAX = 1.0f;
        float MIN = 0.0f;
        for (int i=0; i<n_data; i++) {
            h_d[i] = 1.0f + randomFloat(MIN, MAX);
        }
    } else {
        ERROR("Not implemented!");
    }
    
    CHECK(cudaMemcpy(d_data, h_d, nBytes, cudaMemcpyHostToDevice));
}

void Tensor::initialize(float *h_data, DimVector& shape) {
    assert(this->shape == shape);
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
void Tensor::reshape(DimVector& shape_n) {
    size_t dim = this->getDim();
    size_t dim_n = shape_n.size();

    if(dim_n == dim) {
        // dim swap, internal transpose?
        // TODO
    } else {
        if(isContinuousSqueeze(this->shape, shape_n)) {
            this->shape = shape_n;
        } else {
            ERROR("Failed to reshape!");
            exit(0);
        }
    }
}

void Tensor::transpose() {
    if(this->shape.size() == 2) {
        float* d_out;
        CHECK(cudaMalloc((float**)&d_out, this->n_data * sizeof(float)));
        int x = this->shape[0], y = this->shape[1];
        dim3 block(BLOCK_SIZE2D, BLOCK_SIZE2D);
        // ATTENTION!!!
        dim3 grid((y-1)/BLOCK_SIZE2D + 1, (x-1)/BLOCK_SIZE2D + 1);

        kTranspose<<<grid, block>>>(this->d_data, d_out, x, y);
        CHECK_KERNEL();

        // update this->d_data
        float* prev = this->d_data;
        this->d_data = d_out;
        CHECK(cudaFree(prev));
        std::swap(this->shape[0], this->shape[1]);
    } else {
        ERROR("Transpose failed: the dim != 2.");
    }
}

void Tensor::scale_(float factor) {
    dim3 block(BLOCK_SIZE1D);
    dim3 grid((this->n_data-1)/BLOCK_SIZE1D + 1);
    // need padding?
    kScale<<<grid, block>>>(this->d_data, factor, 0.0f, this->n_data);
    CHECK_KERNEL();
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

float Tensor::mean(){
    float s = sum();
    return s/this->n_data;
}

void Tensor::squeeze() {
    DimVector& vec = this->shape;
    vec.erase(std::remove(vec.begin(), vec.end(), 1), vec.end());
}

void Tensor::squeeze(int idx) {
    DimVector& vec = this->shape;
    if(idx < vec.size() && vec[idx] == 1)
        vec.erase(vec.begin() + idx);
}

/* Binary op */

Tensor* Tensor::saxpy(Tensor* tensor, float f1, float f2) {
    size_t d1 = this->getDim(), d2 = tensor->getDim();
    bool ge = d1 >= d2;
    Tensor& mat1 = ge ? *this : *tensor;
    Tensor& mat2 = ge ? *tensor : *this;
    Tensor* tensor_o = NULL;
    DimVector shape1 = ge ? this->shape : tensor->shape;
    DimVector shape2 = ge ? tensor->shape : this->shape;
    size_t dim1 = shape1.size(), dim2 = shape2.size();

    if(isBroadcast(shape1, shape2)) {
        if(dim1 == 1) {
            // e.x. shape1: (N) shape2: (N)
            assert(shape1[0] == shape2[0]);
            int block = BLOCK_SIZE1D;
            int grid = (shape1[0] - 1) / block + 1;
            DimVector shape_o = {shape[0]};
            tensor_o = new Tensor(shape_o);
            kAdd_l1<<<grid, block>>>((&mat1)->d_data, (&mat2)->d_data, tensor_o->d_data, this->n_data, f1, f2);
        } else if(dim1 == 2) {
            DimVector shape_o = getBroadcastShape(shape1, shape2);
            dim3 block(BLOCK_SIZE2D, BLOCK_SIZE2D);
            int x = shape_o[0], y = shape_o[1];
            dim3 grid((y-1)/BLOCK_SIZE2D+1, (x-1)/BLOCK_SIZE2D+1);
            tensor_o = new Tensor(shape_o);

            if(dim2 == 1) {
            // e.x. shape1: (B x N), shape2: (N) 
                kAdd_l2<<<grid, block>>>((&mat1)->d_data, (&mat2)->d_data, tensor_o->d_data, x, y, f1 ,f2);
            } else {
            // e.x. shape1: (B x N), shape2: (1 x N) 
                int s1 = shape1[0]==1 ? 0 : y;
                int s2 = shape2[0]==1 ? 0 : y;

                kAddStride_l2<<<grid, block>>>((&mat1)->d_data, (&mat2)->d_data, tensor_o->d_data, x, y, f1 ,f2, s1, s2);
            }
            
            
        } else {
            ERROR("Not implemented!");
        }
    } else {
        ERROR("Failed to broadcast!");
    }
    return tensor_o;  
}

Tensor* Tensor::add(Tensor* tensor) {
    return saxpy(tensor, 1.0f, 1.0f);
}

Tensor* Tensor::sub(Tensor* tensor) {
    return saxpy(tensor, 1.0f, -1.0f);
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

Tensor* Tensor::bmm(Tensor* tensor) {
    size_t d1 = this->getDim(), d2 = tensor->getDim();
    bool ge = d1 >= d2;
    Tensor& mat1 = ge ? *this : *tensor;
    Tensor& mat2 = ge ? *tensor : *this;
    DimVector shape1 = ge ? this->shape : tensor->shape;
    DimVector shape2 = ge ? tensor->shape : this->shape;
    size_t dim1 = shape1.size(), dim2 = shape2.size();

    DimVector shape_o = getBatchMatmulShape(shape1, shape2);

    if(!shape_o.empty()) {
        
    } else {
        ERROR("Failed to match mats' shape!");
    }
    return nullptr;
}


/* tool funcs */

bool isContinuousSqueeze(DimVector& shape1, DimVector& shape2) {
    size_t dim1 = shape1.size();
    size_t dim2 = shape2.size();

    int i = dim1 - 1, j = dim2 - 1;
    int acc_1 = 0, acc_2 = 0;

    while(i >= 0 && j >= 0) {
        // printf("shape1[%d] = %ld, shape2[%d] = %ld\n", i, shape1[i], j, shape2[j]);
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
                    printf("acc_2 = %d, shape[%d]=%ld\n", acc_2, i, shape1[i]);
                    ERROR("Shape squeeze invalid!\n");
                }
            } else {
                ERROR("Shape squeeze invalid!\n");
            }
        }
    }

    return i < 0 && j < 0; 
}

bool isBroadcast(DimVector& shape1, DimVector& shape2) {
    int dim1 = shape1.size(), dim2 = shape2.size();
    size_t dim =  dim1 > dim2 ? dim2 : dim1;
    for(int i=1; i<=dim; i++) {
        if(shape1[dim1-i] != shape2[dim2-i]){
            if(shape1[dim1-i] != 1 &&
                shape2[dim2-i] != 1 &&
                shape1[dim-i] != 0 &&
                shape2[dim2-i] != 0) {
                return 0;
            }
        }
    }
    return 1;
}

DimVector getBroadcastShape(DimVector& shape1, DimVector& shape2) {
    int dim1 = shape1.size(), dim2 = shape2.size();
    int dim = dim1 > dim2 ? dim1 : dim2;
    DimVector result_shape = DimVector(dim);
    for(int i=1; i<=dim; i++) {
        if(i <= dim2 && i <= dim1) {
            result_shape[dim-i] = shape2[dim2-i] == 1 ? shape1[dim1-i] : shape2[dim2-i];
        } else if(i<=dim1) {
            result_shape[dim-i] = shape1[dim1-i];
        } else if(i <= dim2) {
            result_shape[dim-i] = shape1[dim2-i];
        }
    }
    return result_shape;
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

