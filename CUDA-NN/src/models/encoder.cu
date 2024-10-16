#include "encoder.cuh"

Encoder::Encoder(std::string prefix, bool global_feat, bool feature_transform, size_t channel) {
    this->global_feat = global_feat;
    this->feature_transform = feature_transform;
    this->channel = channel;

    this->prefix = prefix;

    this->stn = new STN3d(this->prefix + "stn.", channel);
    this->conv1 = new Conv1d(this->prefix + "conv1.", channel, 64, 1);
    this->conv2 = new Conv1d(this->prefix + "conv2.", 64, 128, 1);
    this->conv3 = new Conv1d(this->prefix + "conv3.", 128, 1024, 1);
    this->bn1 = new BatchNorm1d(this->prefix + "bn1.", 64);
    this->bn2 = new BatchNorm1d(this->prefix + "bn2.", 128);
    this->bn3 = new BatchNorm1d(this->prefix + "bn3.", 1024);
    this->relu = new ReLU(this->prefix + "relu.");

    if(feature_transform) {
        fstn = new STNkd(this->prefix + "fstn.", 64);
    }
}

Encoder::~Encoder() {
    delete stn;
    delete conv1;
    delete conv2;
    delete conv3;
    delete bn1;
    delete bn2;
    delete bn3;
    delete relu;
}

void Encoder::load_weights() {
    this->stn->load_weights();
    this->conv1->load_weights();
    this->conv2->load_weights();
    this->conv3->load_weights();
    this->bn1->load_weights();
    this->bn2->load_weights();
    this->bn3->load_weights();
    if(feature_transform) {
        this->fstn->load_weights();
    }
}

Tensor* Encoder::forward(Tensor* data) {
    DimVector shape = data->getShape();
    size_t bz = shape[0], D = shape[1], N = shape[2];

    Tensor* trans = stn->forward(data);

    data->transpose(2, 1);
    assert(D == 3);


    Tensor* x = data->bmm(trans);   // (N L C) @ (N C C)
    x->transpose(2, 1);
    // x = bn1->forward(conv1->forward(x));
    x = relu->forward(bn1->forward(conv1->forward(x)));

    Tensor* trans_feat;

    if(this->feature_transform) {
        trans_feat = fstn->forward(x);
        x->transpose(2, 1);
        Tensor* f_trans = x->bmm(trans_feat);    // track f_trans
        x = f_trans;
        x->transpose(2, 1);
    }

    x = relu->forward(bn2->forward(conv2->forward(x)));
    x = bn3->forward(conv3->forward(x));
    x->max_(2, false);

    /*
    if self.global_feat:
        return x, trans, trans_feat
    */

    // clean up
    // delete trans;
    // if(this->feature_transform) {
    //     delete trans_feat;
    // }

    if(this->global_feat) {
        return x;
    }
    ERROR("Not implemented!\n");
    return nullptr;    
}

Tensor* Encoder::backward(Tensor* gradients) {
    return nullptr;
}

