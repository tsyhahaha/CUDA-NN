#include "pointnet.cuh"

PointNet::PointNet(std::string prefix, size_t k, bool normal_channel){
    size_t channel;
    if (normal_channel) {
        channel = 6;
        ERROR("not implemented!\n");
    } else {
        channel = 3;
    }

    this->prefix = prefix;

    this->feat = new Encoder(this->prefix + "feat.", true, true, channel);
    this->fc1 = new Linear(this->prefix + "fc1.", 1024, 512);
    this->fc2 = new Linear(this->prefix + "fc2.", 512, 256);
    this->fc3 = new Linear(this->prefix + "fc3.", 256, k);
    this->dropout = new Dropout(this->prefix + "dropout.", 0.4);
    this->bn1 = new BatchNorm1d(this->prefix + "bn1.", 512);
    this->bn2 = new BatchNorm1d(this->prefix + "bn2.", 256);
    this->softmax = new SoftMax(this->prefix + "softmax.", 1, true);
    this->relu = new ReLU(this->prefix + "relu.");
}

PointNet::PointNet(size_t k, bool normal_channel) {
    size_t channel;
    if (normal_channel) {
        channel = 6;
        ERROR("not implemented!\n");
    } else {
        channel = 3;
    }

    this->feat = new Encoder(this->prefix + "feat.", true, true, channel);
    this->fc1 = new Linear(this->prefix + "fc1.", 1024, 512);
    this->fc2 = new Linear(this->prefix + "fc2.", 512, 256);
    this->fc3 = new Linear(this->prefix + "fc3.", 256, k);
    this->dropout = new Dropout(this->prefix + "dropout.", 0.4);
    this->bn1 = new BatchNorm1d(this->prefix + "bn1.", 512, true);
    this->bn2 = new BatchNorm1d(this->prefix + "bn2.", 256, true);
    this->softmax = new SoftMax(this->prefix + "softmax.", 1, true);
    this->relu = new ReLU(this->prefix + "relu.");
}

PointNet* PointNet::train() {
    this->is_training = true;
    this->feat->train();
    this->fc1->train();
    this->fc2->train();
    this->fc3->train();
    this->bn1->train();
    this->bn2->train();
    softmax->train();
    return this;
}

PointNet* PointNet::eval() {
    this->is_training = false;
    this->feat->eval();
    this->fc1->eval();
    this->fc2->eval();
    this->fc3->eval();
    this->bn1->eval();
    this->bn2->eval();
    return this;
}

void PointNet::load_weights() {
    this->feat->load_weights();
    this->fc1->load_weights();
    this->fc2->load_weights();
    this->fc3->load_weights();
    this->bn1->load_weights();
    this->bn2->load_weights();
}

void PointNet::init_weights() {
    this->feat->init_weights();
    this->fc1->init_weights();
    this->fc2->init_weights();
    this->fc3->init_weights();
    this->bn1->init_weights();
    this->bn2->init_weights();
}

PointNet::~PointNet(){
    delete feat;
    delete fc1;
    delete fc2;
    delete fc3;
    delete dropout;
    delete bn1;
    delete bn2;
    delete softmax;
    delete relu;
}

void PointNet::name_params(std::map<std::string, Tensor*>& name_params) {
    this->fc1->name_params(name_params);
    this->fc2->name_params(name_params);
    this->fc3->name_params(name_params);
    this->bn1->name_params(name_params);
    this->bn2->name_params(name_params);
    this->feat->name_params(name_params);
}

Tensor* PointNet::forward(Tensor* data, Tensor* mask) {
    Tensor* x = feat->forward(data, mask);
    x = bn1->forward(fc1->forward(x));
    x = fc2->forward(x);
    // if(this->is_training) {
    //     x = dropout->forward(x);
    // }
    x = bn2->forward(x);
    x = fc3->forward(x);     // (N x num_classes)
    x = softmax->forward(x);
    return x;
}

Tensor* PointNet::backward(Tensor* gradients) {
    Tensor* d_o = softmax->backward(gradients);
    d_o = fc3->backward(d_o);
    d_o = fc2->backward(bn2->backward(d_o));
    d_o = fc1->backward(bn1->backward(d_o));
    d_o = feat->backward(d_o);

    return d_o;
}

