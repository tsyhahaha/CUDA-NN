#include "linear.cuh"

Linear::Linear(size_t in_features, size_t out_features, bool bias = 0, InitType init_type = ZERO) {
    this->in_features = in_features;
    this->out_features = out_features;

    DimVector weight_shape = {in_features, out_features};
    this->weight = new Tensor(weight_shape);

    this->weight->initialize(init_type);
    
    if(bias) {
        DimVector bias_shape = {out_features};
        this->bias = new Tensor(bias_shape, ZERO);
        this->bias->initialize(0.0f);
    }
}

void Linear::load_weights(float *h_weight_data, float *h_bias_data, DimVector weight_shape, DimVector bias_shape) {
    assert(this->weight->getShape() == weight_shape);
    this->weight->initialize(h_weight_data, weight_shape);        
    assert(this->bias->getShape() == bias_shape);
    this->bias->initialize(h_bias_data, bias_shape);
}

void Linear::load_weights(float *h_data, DimVector shape, const std::string& target) {
    if(target == "weight") {
        assert(this->weight->getShape() == shape);
        this->weight->initialize(h_data, shape);        
    } else if(target == "bias") {
        assert(this->bias->getShape() == shape);
        this->bias->initialize(h_data, shape);
    }
}

Tensor* Linear::forward(Tensor* data) {
    // weights(N x M) * data(B x N) + bias(M x 1) = output(B x M)
    this->input = data;


    Tensor* mul_o = data->matmul(this->weight);  // keep the batch dim at the first dimension


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

    assert(this->output->getShape()[0] == input->getShape()[0] && \
            this->output->getShape()[1] == weight->getShape()[1]);

    return this->output;
}

Tensor* Linear::backward(Tensor* gradients) {
    return nullptr;
}