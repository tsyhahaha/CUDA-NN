#include "pointnet.cuh"
#include "configure.cuh"

PointNet::PointNet(std::string prefix, size_t k, bool normal_channel) {
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
    this->softmax = new SoftMax(this->prefix + "softmax.", 1);
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
    this->softmax = new SoftMax(this->prefix + "softmax.", 1);
    this->relu = new ReLU(this->prefix + "relu.");
}

void PointNet::load_weights() {
    this->feat->load_weights();
    this->fc1->load_weights();
    this->fc2->load_weights();
    this->fc3->load_weights();
    this->bn1->load_weights();
    this->bn2->load_weights();
}

Tensor* PointNet::forward(Tensor* data) {
    Tensor* x = feat->forward(data);
    x = bn1->forward(fc1->forward(x));

    x = bn2->forward(fc2->forward(x));
    x = fc3->forward(x);     // (B x num_classes)
    // x = softmax->forward(x);

    return x;
}

Tensor* PointNet::backward(Tensor* gradients) {
    return nullptr;
}

