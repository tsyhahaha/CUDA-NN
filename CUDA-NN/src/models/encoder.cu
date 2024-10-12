#include "encoder.cuh"

Encoder::Encoder(bool global_feat, bool feature_transform, size_t channel) {
    this->global_feat = global_feat;
    this->feature_transform = feature_transform;
    this->channel = channel;

    this->stn = new STN3d(channel);
    this->conv1 = new Conv1d(channel, 64, 1);
    this->conv2 = new Conv1d(64, 128, 1);
    this->conv3 = new Conv1d(128, 1024, 1);
    this->bn1 = new BatchNorm1d(64);
    this->bn2 = new BatchNorm1d(128);
    this->bn3 = new BatchNorm1d(1024);
    this->relu = new ReLU();

    if(feature_transform) {
        fstn = new STNkd(64);
    }

}

Encoder::~Encoder() {
    delete stn, conv1, conv2, conv3, bn1, bn2, bn3, relu;
}

Tensor* Encoder::forward(Tensor* data) {
    DimVector shape = data->getShape();
    size_t bz = shape[0], D = shape[1], N = shape[2];

    Tensor* trans = stn->forward(data);

    data->transpose(2, 1);
    assert(D == 3);

    Tensor* data_trans = data->bmm(trans);   // (N L C) @ (N C C)
    Tensor* x = relu->forward(bn1->forward(conv1->forward(data_trans)));

    if(this->feature_transform) {
        Tensor* trans_feat = fstn->forward(x);
        x->transpose(-1, -2);
        Tensor* f_trans = x->bmm(trans_feat);    // track f_trans
        x = f_trans;
        x->transpose(-1, -2);
    } else {
        Tensor* trans_feat = nullptr;
    }


    x = relu->forward(bn2->forward(conv2->forward(x)));
    x = bn3->forward(conv3->forward(x));
    x->max_(2, false);

    /*
    if self.global_feat:
        return x, trans, trans_feat
    */

    if(this->global_feat) {
        return x;
    }
    return nullptr;    
}

Tensor* Encoder::backward(Tensor* gradients) {
    return nullptr;
}