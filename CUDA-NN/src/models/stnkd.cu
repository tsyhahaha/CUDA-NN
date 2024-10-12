#include "stnkd.cuh"

STNkd::STNkd(size_t k) {
    this->k = k;

    this->conv1 = new Conv1d(k, 64, 1);
    this->conv2 = new Conv1d(64, 128, 1);
    this->conv3 = new Conv1d(128, 1024, 1);

    this->fc1 = new Linear(1024, 512);
    this->fc2 = new Linear(512, 256);
    this->fc3 = new Linear(256, k * k);
    this->relu = new ReLU();

    this->bn1 = new BatchNorm1d(64);
    this->bn2 = new BatchNorm1d(128);
    this->bn3 = new BatchNorm1d(1024);
    this->bn4 = new BatchNorm1d(512);
    this->bn5 = new BatchNorm1d(256);
}

STNkd::~STNkd() {
    delete conv1, conv2, conv3, fc1, fc2, fc3, relu, bn1, bn2, bn3, bn4, bn5;
}

Tensor* STNkd::forward(Tensor* data) {
    size_t bz = data->getShape()[0];
    Tensor* x = bn1->forward(conv1->forward(data));
    x = bn2->forward(conv2->forward(x));
    x = bn3->forward(conv3->forward(x));
    x->max_(-1, false);

    x = relu->forward(bn4->forward(fc1->forward(x)));
    x = relu->forward(bn5->forward(fc2->forward(x)));
    x = fc3->forward(x);

    Tensor* iden = new Tensor({k,k}, IDENTITY);
    iden->flatten();

    Tensor* o = x->add(iden);
    o->reshape({bz, k, k});

    delete x;
    return o;
}

Tensor* STNkd::backward(Tensor* gradients) {
    return nullptr;
}