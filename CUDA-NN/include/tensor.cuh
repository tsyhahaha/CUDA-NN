#pragma once
#ifndef TENSOR_H
#define TENSOR_H

#include"kernels.cuh"
#include "configure.cuh"
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
        size_t n_data = 0;
        DimVector shape;

        float* transpose_cache = nullptr;
        float* mask_cache = nullptr;

        bool is_training = false;
        Tensor* gradients_acc = nullptr;

    public:
        static void oneHot(int* d_labels, Tensor*& target, size_t label_num, size_t class_num, cudaStream_t stream);

    public:
        Tensor(DimVector shape, InitType init_type = NONE);
        Tensor(float *h_data, DimVector shape);
        Tensor(const Tensor &other);
        Tensor(std::vector<float>& data_vec, DimVector shape);
        ~Tensor();
        bool is_train();
        void train();
        void eval();

        bool checkNan(bool check_grad = false);

        void fromVec(std::vector<float>& vec);
        void copyFrom(Tensor* x);
        void reset(DimVector shape);
        std::vector<float> toVec();

        // getters
        size_t getDim();
        float* toHost();
        float* getData();
        size_t getSize();
        size_t getSize(int dim);
        size_t getMemSize();
        DimVector getShape();

        Tensor* getGradsAcc();

        // setters
        void setShape(DimVector shape);
        void setData(float* data);

        // set value
        void load(float* h_data, size_t n_data);
        void initialize(float value);
        void initialize(InitType type, float bound = 1.0f);
        void initialize(float *h_data, DimVector& shape);

        // shape
        Tensor* squeeze();
        Tensor* squeeze(int idx);
        Tensor* unsqueeze(int idx = 0);

        // training/inference
        void acc_grads(Tensor* grads);

        // self transform
        void transpose(int d1, int d2);
        void transpose();
        void reshape(DimVector n_shape);
        void flatten();

        // Unary op
        Tensor* scale_(float factor);
        void add_(float c);
        void sub_(float c);
        void add_(Tensor* tensor, float f1=1.0f, float f2=1.0f);
        void sub_(Tensor* tensor, float f1=1.0f, float f2=1.0f);
        void mask_fill_(Tensor*& mask, int dim, float value);
        Tensor* mask(Tensor* ref);

        float sum();
        Tensor* sum(Tensor*& tensor_o, int dim);
        // Tensor* sum(Tensor*& tensor_o, int dim, bool keepDim=true);
        Tensor* sumToDim(Tensor*& tensor_o, int dim);
        Tensor* sumToDim_(int dim);
        float mean();
        Tensor* mean_(int dim, bool keepDim=false);
        Tensor* mean(Tensor*& tensor_o, int dim, bool keepDim=false);
        Tensor* var(Tensor*& tensor_o, int dim, Tensor* mean = nullptr, bool keepDim=false);
        Tensor* max(int dim, bool keepDim=true);
        Tensor* max(Tensor*& output, int dim, bool keepDim=true);
        Tensor* max_wt_index(Tensor*& output, Tensor*& max_index, int dim, bool keepDim=true);
        Tensor* argmax(int dim, bool keepDim=true);
        void argmax(Tensor*& output, int dim, bool keepDim=true);
        void max_(int dim, bool keepDim=true);
        // Tensor* sum(size_t d);
        // Tensor* mean(size_t d);
        Tensor* exp();
        Tensor* log();
        void exp_();
        void log_();
        void square_();

        // Binary op
        Tensor* add(Tensor*& output, Tensor* tensor);
        Tensor* sub(Tensor*& output, Tensor* tensor);
        Tensor* matmul(Tensor*& output, Tensor* tensor);
        Tensor* dot(Tensor*& output, Tensor* tensor, float factor = 1.0f);
        Tensor* div(Tensor*& output, Tensor* tensor, float factor = 1.0f);
        Tensor* bmm(Tensor* tensor);
        void bmm(Tensor*& output, Tensor* tensor);
    private:
        Tensor* saxpy(Tensor*& output, Tensor* tensor, float f1, float f2);
        Tensor* saxpy_plus(Tensor*& output, Tensor* tensor, float factor, int flag);
        void saxpy_(Tensor* tensor, float f1, float f2);
};

#endif /* !TENSOR_H */