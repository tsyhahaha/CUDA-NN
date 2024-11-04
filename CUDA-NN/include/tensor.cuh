#pragma once
#ifndef TENSOR_H
#define TENSOR_H

#include"kernels.cuh"
#include<math.h>
#include<assert.h>
#include<iostream>
#include<vector>
#include<algorithm>
#include<numeric>


#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])
typedef std::vector<size_t> DimVector;

enum InitType {
    NONE, ZERO, ONES, IDENTITY, RANDOM, XAVIER, KAIMING
};

class Tensor {
    private:
        float *d_data;
        size_t n_data;
        DimVector shape;

        float* transpose_cache = nullptr;
        float* mask_cache = nullptr;

    public:
        Tensor(DimVector shape, InitType init_type = NONE);
        Tensor(float *h_data, DimVector shape);
        Tensor(const Tensor &other);
        ~Tensor();
        void fromVec(std::vector<float>& vec);
        void copyFrom(Tensor* x);
        void reset(DimVector shape);
        std::vector<float> toVec();

        // getters
        size_t getDim();
        size_t getSize(size_t dim);
        float* toHost();
        float* getData();
        size_t getSize();
        DimVector getShape();

        // setters
        void setShape(DimVector shape);
        void setData(float* data);

        // set value
        void load(float* h_data, size_t n_data);
        void initialize(float value);
        void initialize(InitType type);
        void initialize(float *h_data, DimVector& shape);

        // shape
        void squeeze();
        void squeeze(int idx);
        void unsqueeze(int idx = 0);

        // self transform
        void transpose(int d1, int d2);
        void transpose();
        void reshape(DimVector n_shape);
        void flatten();

        // Unary op
        void scale_(float factor);
        void add_(float c);
        void sub_(float c);
        void add_(Tensor* tensor);
        void sub_(Tensor* tensor);
        void mask_fill_(Tensor*& mask, int dim, float value);
        // Tensor* scale(float factor);
        float sum();
        Tensor* sum(int dim);
        float mean();
        Tensor* max(int dim, bool keepDim=true);
        Tensor* max(Tensor*& output, int dim, bool keepDim = true);
        Tensor* argmax(int dim, bool keepDim=true);
        void argmax(Tensor*& output, int dim, bool keepDim=true);
        void max_(int dim, bool keepDim=true);
        // Tensor* sum(size_t d);
        // Tensor* mean(size_t d);
        Tensor* exp();
        Tensor* log();
        void exp_();
        void log_();

        // Binary op
        Tensor* add(Tensor* tensor);
        Tensor* sub(Tensor* tensor);
        Tensor* matmul(Tensor* tensor);
        Tensor* dot(Tensor* tensor, float factor = 1.0f);
        Tensor* div(Tensor* tensor, float factor = 1.0f);
        Tensor* bmm(Tensor* tensor);
        void bmm(Tensor*& output, Tensor* tensor);
    private:
        Tensor* saxpy(Tensor* tensor, float f1, float f2);
        Tensor* saxpy_plus(Tensor* tensor, float factor, int flag);
        void saxpy_(Tensor* tensor, float f1, float f2);
};

#endif /* !TENSOR_H */