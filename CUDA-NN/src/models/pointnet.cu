#include "pointnet.cuh"

PointNet::PointNet(size_t k, bool normal_channel) {
    size_t channel;
    if (normal_channel) {
        channel = 6;
        // ERROR("not implemented!\n");
    } else {
        channel = 3;
    }

    this->feat = new Encoder(true, true, channel);
    this->fc1 = new Linear(1024, 512);
    this->fc2 = new Linear(512, 256);
    this->fc3 = new Linear(256, k);
    this->dropout = new Dropout(0.4);
    this->bn1 = new BatchNorm1d(512);
    this->bn2 = new BatchNorm1d(256);
    this->softmax = new SoftMax(1);
    this->relu = new ReLU();
}

Tensor* PointNet::forward(Tensor* data) {
    Tensor* x = feat->forward(data);
    DEBUG_PRINT("finish feat trace\n");
    x = relu->forward(bn1->forward(fc1->forward(x)));
    // x = F.relu(self.bn2(self.dropout(self.fc2(x))))
    x = relu->forward(bn2->forward(fc2->forward(x)));
    x = fc3->forward(x);     // (B x num_classes)
    x = softmax->forward(x);

    return x;
}

Tensor* PointNet::backward(Tensor* gradients) {
    return nullptr;
}