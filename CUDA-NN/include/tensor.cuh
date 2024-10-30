#pragma once
#ifndef TENSOR_H
#define TENSOR_H

#include "kernels.cuh"
#include<math.h>
#include<assert.h>
#include<iostream>
#include<vector>
#include<algorithm>

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

    public:
        Tensor(DimVector shape, InitType init_type = NONE);
        Tensor(float *h_data, DimVector shape);
        ~Tensor();
        void fromVec(std::vector<float>& vec);
        std::vector<float> toVec();

        // getters
        size_t getDim();
        size_t getSize(size_t dim);
        float* toHost();
        float* getData();
        size_t getDataNum();
        DimVector getShape();

        // set value
        void load(float* h_data, size_t n_data);
        void initialize(float value);
        void initialize(InitType type);
        void initialize(float *h_data, DimVector& shape);

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
        void sqrt_();
        void add_(float c);
        void sub_(float c);
        // Tensor* scale(float factor);
        float sum();
        Tensor* sum(int dim);
        float mean();
        Tensor* max(int dim, bool keepDim=true);
        Tensor* argmax(int dim, bool keepDim=true);
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
    private:
        Tensor* saxpy(Tensor* tensor, float f1, float f2);
        Tensor* saxpy_plus(Tensor* tensor, float factor, int flag);
        // Tensor* saxpy_(Tensor* tensor, float f1, float f2);
};

#endif /* !TENSOR_H */