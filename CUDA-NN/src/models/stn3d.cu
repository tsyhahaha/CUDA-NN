#include "stn3d.cuh"

STN3d::STN3d(std::string prefix, size_t channel) {
    this->prefix = prefix;
    
    this->conv1 = new Conv1d(this->prefix + "conv1.", channel, 64, 1);
    this->conv2 = new Conv1d(this->prefix + "conv2.", 64, 128, 1);
    this->conv3 = new Conv1d(this->prefix + "conv3.", 128, 1024, 1);

    this->fc1 = new Linear(this->prefix + "fc1.", 1024, 512);
    this->fc2 = new Linear(this->prefix + "fc2.", 512, 256);
    this->fc3 = new Linear(this->prefix + "fc3.", 256, 9);
    this->relu = new ReLU(this->prefix + "relu.");

    this->bn1 = new BatchNorm1d(this->prefix + "bn1.", 64, true);
    this->bn2 = new BatchNorm1d(this->prefix + "bn2.", 128, true);
    this->bn3 = new BatchNorm1d(this->prefix + "bn3.", 1024, true);
    this->bn4 = new BatchNorm1d(this->prefix + "bn4.", 512, true);
    this->bn5 = new BatchNorm1d(this->prefix + "bn5.", 256, true);
}

STN3d::~STN3d() {
    delete conv1, conv2, conv3, fc1, fc2, fc3, relu, bn1, bn2, bn3, bn4, bn5;
}

void STN3d::load_weights() {
    this->conv1->load_weights();
    this->conv2->load_weights();
    this->conv3->load_weights();

    this->fc1->load_weights();
    this->fc2->load_weights();
    this->fc3->load_weights();

    this->bn1->load_weights();
    this->bn2->load_weights();
    this->bn3->load_weights();
    this->bn4->load_weights();
    this->bn5->load_weights();
}

Tensor* STN3d::forward(Tensor* data) {
    size_t bz = data->getShape()[0];
    Tensor* x = bn1->forward(conv1->forward(data));
    x = bn2->forward(conv2->forward(x));

    x = bn3->forward(conv3->forward(x));
    x->max_(2, false);

    x = bn4->forward(fc1->forward(x));
    x = bn5->forward(fc2->forward(x));
    x = fc3->forward(x);

    Tensor* iden = new Tensor({3,3}, IDENTITY);
    iden->flatten();

    Tensor* o = x->add(iden);
    o->reshape({bz, 3, 3});

    delete iden;

    return o;

}

Tensor* STN3d::backward(Tensor* gradients) {
    return nullptr;
}