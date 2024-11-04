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

    this->iden = new Tensor({3,3}, IDENTITY);
    this->iden->flatten();

    this->o = new Tensor({Configurer::batch_size, 1024});
    this->output = new Tensor({Configurer::batch_size, 9});
}

STN3d::~STN3d() {
    delete conv1;
    delete conv2;
    delete conv3;
    delete fc1;
    delete fc2;
    delete fc3;
    delete relu;
    delete bn1;
    delete bn2;
    delete bn3;
    delete bn4;
    delete bn5;
    
    delete iden;
    if(o != nullptr) delete o;
    if(output != nullptr) delete output;
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

Tensor* STN3d::forward(Tensor* data, Tensor* mask) {
    size_t bz = data->getShape()[0];
    Tensor* x = bn1->forward(conv1->forward(data));
    x = bn2->forward(conv2->forward(x));
    x = bn3->forward(conv3->forward(x));
    if(mask->getSize() > 0) { 
        x->mask_fill_(mask, 2, FP32_MIN);
    }
    
    x->max(o, 2, false);    // the output of max saved to tensor o;

    x = bn4->forward(fc1->forward(o));
    x = bn5->forward(fc2->forward(x));
    x = fc3->forward(x);
    
    // if(output != nullptr) {
    //     output->setShape({bz, 9});
    // } else output = new Tensor({bz, 9});
    // output->copyFrom(x);

    // output->add_(iden);
    // output->reshape({bz, 3, 3});

    x->add_(iden);
    x->reshape({bz, 3, 3});

    return x;
}

Tensor* STN3d::backward(Tensor* gradients) {
    return nullptr;
}