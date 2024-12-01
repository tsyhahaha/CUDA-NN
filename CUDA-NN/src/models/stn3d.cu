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
    this->max_index = new Tensor({Configurer::batch_size, 1024});
    
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
    if(max_index != nullptr) delete max_index;
    if(max_gradients!=nullptr) delete max_gradients;
    if(output != nullptr) delete output;
}

STN3d* STN3d::eval() {
    this->is_training = false;
    this->conv1->eval();
    this->conv2->eval();
    this->conv3->eval();

    this->fc1->eval();
    this->fc2->eval();
    this->fc3->eval();

    this->bn1->eval();
    this->bn2->eval();
    this->bn3->eval();
    this->bn4->eval();
    this->bn5->eval();
    return this;
}

STN3d* STN3d::train() {
    this->is_training = true;
    this->conv1->train();
    this->conv2->train();
    this->conv3->train();

    this->fc1->train();
    this->fc2->train();
    this->fc3->train();

    this->bn1->train();
    this->bn2->train();
    this->bn3->train();
    this->bn4->train();
    this->bn5->train();

    this->max_gradients = new Tensor({Configurer::batch_size, 1024, Configurer::cropping_size});
    return this;
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

void STN3d::init_weights() {
    this->conv1->init_weights();
    this->conv2->init_weights();
    this->conv3->init_weights();

    this->fc1->init_weights();
    this->fc2->init_weights();
    this->fc3->init_weights();

    this->bn1->init_weights();
    this->bn2->init_weights();
    this->bn3->init_weights();
    this->bn4->init_weights();
    this->bn5->init_weights();
}

void STN3d::name_params(std::map<std::string, Tensor*>& np) {
    this->conv1->name_params(np);
    this->conv2->name_params(np);
    this->conv3->name_params(np);
    this->fc1->name_params(np);
    this->fc2->name_params(np);
    this->fc3->name_params(np);
    this->bn1->name_params(np);
    this->bn2->name_params(np);
    this->bn3->name_params(np);
    this->bn4->name_params(np);
    this->bn5->name_params(np);
}

Tensor* STN3d::forward(Tensor* data, Tensor* mask) {
    this->input = data;
    this->mask = mask;

    size_t bz = data->getShape()[0];
    Tensor* x = bn1->forward(conv1->forward(data));
    // return x;
    x = bn2->forward(conv2->forward(x));
    x = bn3->forward(conv3->forward(x));

    if(mask->getSize() > 0) { 
        x->mask_fill_(mask, 2, FP32_MIN);
    }
    
    if(this->is_training) {
        x->max_wt_index(o, max_index, 2, false); 
    } else {
        x->max(o, 2, false);
    }

    x = bn4->forward(fc1->forward(o));
    x = bn5->forward(fc2->forward(x));
    x = fc3->forward(x);

    x->add_(iden);
    x->reshape({bz, 3, 3});

    return x;
}

Tensor* STN3d::backward(Tensor* gradients) {
    size_t N = gradients->getSize(0), C = 1024, L = Configurer::cropping_size;
    gradients->reshape({N, 9});
    Tensor* g = fc3->backward(gradients);
    g = fc2->backward(bn5->backward(g));
    g = fc1->backward(bn4->backward(g));

    dim3 block(BLOCK_SIZE3D, BLOCK_SIZE3D, BATCH_BASE);
    dim3 grid((L-1)/BLOCK_SIZE3D+1, (C-1)/BLOCK_SIZE3D+1, (N-1)/BATCH_BASE+1);

    max_gradients->reset({N, C, L});
    kMaxBackprop<<<grid, block>>>(g->getData(), max_gradients->getData(), max_index->getData(), N, C, L); CHECK_KERNEL();

    g = conv3->backward(bn3->backward(max_gradients));
    g = conv2->backward(bn2->backward(g));
    g = conv1->backward(bn1->backward(g));

    return g;
}