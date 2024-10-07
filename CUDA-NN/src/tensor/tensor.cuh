#pragma once
#ifndef TENSOR_H
#define TENSOR_H

#include "kernels/kernels.cuh"
#include<iostream>
#include<vector>
#include<algorithm>

typedef std::vector<size_t> DimVector;

enum InitType {
    ZERO, ONES, IDENTITY, RANDOM, XAVIER, KAIMING
};

class Tensor {
    private:
        float *d_data;
        size_t n_data;
        DimVector shape;

    public:
        Tensor(DimVector& shape);
        Tensor(float *h_data, DimVector& shape);
        ~Tensor();

        // getters
        size_t getDim();
        size_t getSize(size_t dim);
        float* toHost();
        float* getData();
        size_t getDataNum();
        DimVector getShape();

        // set value
        void initialize(float value);
        void initialize(InitType type);
        void initialize(float *h_data, DimVector shape);

        void squeeze();
        void squeeze(int idx);

        // self transform
        // void transpose(size_t d1, size_t d2);
        void transpose();
        // void reshape(size_t *n_shape, size_t n_dim);

        // Unary op
        void scale_(float factor);
        // Tensor* scale(float factor);
        float sum();
        float mean();
        // Tensor* sum(size_t d);
        // Tensor* mean(size_t d);

        // Binary op
        Tensor* add(Tensor* tensor);
        Tensor* sub(Tensor* tensor);
        Tensor* matmul(Tensor* tensor);
        Tensor* bmm(Tensor* tensor);
    private:
        Tensor* saxpy(Tensor* tensor, float f1, float f2);
        // Tensor* saxpy_(Tensor* tensor, float f1, float f2);
};

#endif /* !TENSOR_H */