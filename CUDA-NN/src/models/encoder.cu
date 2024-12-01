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
    this->bn1 = new BatchNorm1d(this->prefix + "bn1.", 64, true);
    this->bn2 = new BatchNorm1d(this->prefix + "bn2.", 128, true);
    this->bn3 = new BatchNorm1d(this->prefix + "bn3.", 1024);
    this->relu = new ReLU(this->prefix + "relu.");

    if(feature_transform) {
        fstn = new STNkd(this->prefix + "fstn.", 64);
    }

    this->p_trans = new Tensor({Configurer::batch_size, Configurer::cropping_size, channel});
    this->f_trans = new Tensor({Configurer::batch_size, Configurer::cropping_size, 64});
    this->output = new Tensor({Configurer::batch_size, 1024});

    this->max_index = new Tensor({Configurer::batch_size, 1024});
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

    if(p_trans != nullptr) delete p_trans;
    if(f_trans != nullptr) delete f_trans;
    if(output != nullptr) delete output;

    if(max_index!=nullptr) delete max_index;
    if(max_gradients != nullptr) delete max_gradients;
    if(!f_gradients) delete f_gradients;
    if(!trans_feat_gradients) delete trans_feat_gradients;
    if(!trans_points_gradients) delete trans_points_gradients;
    if(!p_gradients) delete p_gradients;
}


Encoder* Encoder::train() {
    this->is_training = true;
    this->stn->train();
    this->conv1->train();
    this->conv2->train();
    this->conv3->train();
    this->bn1->train();
    this->bn2->train();
    this->bn3->train();
    if(feature_transform) {
        this->fstn->train();
    }
    this->max_gradients = new Tensor({Configurer::batch_size, 1024, Configurer::cropping_size});
    this->trans_feat_gradients = new Tensor({Configurer::batch_size, 64, 64});
    this->f_gradients = new Tensor({Configurer::batch_size, 64, Configurer::cropping_size});
    this->trans_points_gradients = new Tensor({Configurer::batch_size, 3, 3});
    return this;
}

Encoder* Encoder::eval() {
    this->is_training = false;
    this->stn->eval();
    this->conv1->eval();
    this->conv2->eval();
    this->conv3->eval();
    this->bn1->eval();
    this->bn2->eval();
    this->bn3->eval();
    if(feature_transform) {
        this->fstn->eval();
    }
    return this;
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

void Encoder::init_weights() {
    this->stn->init_weights();
    this->conv1->init_weights();
    this->conv2->init_weights();
    this->conv3->init_weights();
    this->bn1->init_weights();
    this->bn2->init_weights();
    this->bn3->init_weights();
    if(feature_transform) {
        this->fstn->init_weights();
    }
}

void Encoder::name_params(std::map<std::string, Tensor*>& np) {
    this->stn->name_params(np);
    this->conv1->name_params(np);
    this->conv2->name_params(np);
    this->conv3->name_params(np);
    this->bn1->name_params(np);
    this->bn2->name_params(np);
    this->bn3->name_params(np);
    if(feature_transform) {
        this->fstn->name_params(np);
    }
}

Tensor* Encoder::forward(Tensor* data, Tensor* mask) {
    this->input = data;
    this->mask = mask;

    DimVector shape = data->getShape();
    size_t bz = shape[0], D = shape[1], N = shape[2];

    trans_points = stn->forward(data, mask);    // (N, 3, 3)
    data->transpose(2, 1);  // (N, 3, L)->(N, L, 3)
    assert(D == 3);

    data->bmm(p_trans, trans_points);   // (N, L, 3) @ (N, 3, 3)->(N, L, 3)
    p_trans->transpose(2, 1);           // (N, L, 3)->(N, 3, L)

    Tensor* x;
    feat = bn1->forward(conv1->forward(p_trans));  //(N, 64, L)
    // this->feat->copyFrom(x);

    if(this->feature_transform) {
        trans_feat = fstn->forward(feat, mask);
        feat->transpose(2, 1);     // (N, L, 64)
        feat->bmm(f_trans, trans_feat);    // (N, L, 64) @ (N, 64, 64)->(N, L, 64)
        feat->transpose(2, 1);
        f_trans->transpose(2, 1);       // (N, L, 64) -> (N, 64, L)
        x = f_trans;
    }


    x = bn2->forward(conv2->forward(x));
    x = bn3->forward(conv3->forward(x));    // the output has been changed, attention when impl training

    if(mask->getSize() > 0) {
        x->mask_fill_(mask, 2, FP32_MIN);
    }

    x->max_wt_index(output, max_index, 2, false);   // saved to the output

    /*
    if self.global_feat:
        return x, trans, trans_feat
    */

    if(this->global_feat) {
        return output;
    }
    ERROR("Not implemented!\n");
    return nullptr;    
}

Tensor* Encoder::backward(Tensor* gradients) {
    size_t N = gradients->getSize(0), C = gradients->getSize(1), L = Configurer::cropping_size;
    dim3 block(BLOCK_SIZE3D, BLOCK_SIZE3D, BATCH_BASE);
    dim3 grid((L-1)/BLOCK_SIZE3D+1, (C-1)/BLOCK_SIZE3D+1, (N-1)/BATCH_BASE+1);
    max_gradients->reset({N, C, L});
    kMaxBackprop<<<grid, block>>>(gradients->getData(), max_gradients->getData(), max_index->getData(), N, C, L); CHECK_KERNEL();

    Tensor* g = conv3->backward(bn3->backward(max_gradients));
    g = conv2->backward(bn2->backward(g));  // f_trans_gradient:(N, 64, L)

    if(feature_transform) {
        f_gradients->reset({N, 64, L});
        trans_feat_gradients->reset({N, 64, 64});

        trans_feat->bmm(f_gradients, g);    // (N, 64, 64) x (N, 64, L)->(N, 64, L)

        g->transpose(2, 1);   // (N, 64, L) -> (N, L, 64)
        feat->bmm(trans_feat_gradients, g);  // (N, 64, L) x (N, L, 64)->(N, 64, 64)

        g = fstn->backward(trans_feat_gradients);   // (N, 64, L)
        g->add_(f_gradients);
    }
    g = conv1->backward(bn1->backward(g));  // (N, 3, L)
    g->bmm(trans_points_gradients, input); // (N, 3, L) x (N, L, 3)->(N, 3, 3)
    stn->backward(trans_points_gradients);

    return nullptr; // no need to backward on input points
}

