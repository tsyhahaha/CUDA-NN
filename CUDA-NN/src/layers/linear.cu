#include "linear.cuh"

Linear::Linear(std::string prefix, size_t in_features, size_t out_features, bool bias, InitType init_type) {
    this->in_features = in_features;
    this->out_features = out_features;

    DimVector weights_shape = {out_features, in_features};
    this->weights = new Tensor(weights_shape);

    this->weights->initialize(init_type);

    this->prefix = prefix;
    
    if(bias) {
        DimVector bias_shape = {out_features};
        this->bias = new Tensor(bias_shape, ZERO);
        this->bias->initialize(0.0f);
    }
}

Linear::~Linear() {
    delete weights, bias, input, output, outputBackward;
}

Tensor* Linear::forward(Tensor* data) {
    // data(B x N) @ weightss(M x N).T + bias(M) = output(B x M)
    this->input = data;


    Tensor* mul_o = data->matmul(this->weights);  // keep the batch dim at the first dimension


    if(this->bias){
        this->output = mul_o->add(this->bias);
        delete mul_o;
    }
    else {
        this->output = mul_o;
    }

    /////////////
    //DEBUG_PRINT
    /////////////

    if(this->output->getShape()[0] != input->getShape()[0] ||  \
            this->output->getShape()[1] != weights->getShape()[0]) {
                printShape(this->output->getShape());
                ERROR("shape not matched!\n");
            }

    return this->output;
}

Tensor* Linear::backward(Tensor* gradients) {
    return nullptr;
}