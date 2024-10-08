#include "tensor.cuh"
#include "conv1d.cuh"

Conv1d::Conv1d(size_t in_channels, size_t out_channels, size_t kernel_size, bool bias) {
    this->in_channels = in_channels;
    this->out_channels = out_channels;
    this->kernel_size = this->kernel_size;

    if(kernel_size != 1) {
        perror("Not implemented!");
    }

    if(bias) {
        DimVector shape_bias = {(size_t)out_channels};
        this->bias = new Tensor(shape_bias, ZERO);
    }

    DimVector shape_weight = {(size_t)out_channels, (size_t)in_channels};
    this->weight = new Tensor(shape_weight, RANDOM);
}


void Conv1d::load_weights(float* h_data, DimVector& shape, const std::string& target) {
    if(target == "weight") {
        assert(this->weight->getShape() == shape);
        this->weight->initialize(h_data, shape);
    } else if(target == "bias") {
        assert(this->bias->getShape() == shape);
        this->bias->initialize(h_data, shape);
    }
}

void Conv1d::load_weights(float *h_weight_data, float *h_bias_data, DimVector weight_shape, DimVector bias_shape) {
    assert(this->weight->getShape() == weight_shape);
    this->weight->initialize(h_weight_data, weight_shape);        
    assert(this->bias->getShape() == bias_shape);
    this->bias->initialize(h_bias_data, bias_shape);
}

/*
im2col to accelerate conv op
 - input: (B x N x in_channels) -> (B*N x in_channels)
 - weights: (out_channels x in_channels)
 - bias: (out_channels)
 - output(reshape): (B*N out_channels) + bias(out_channels) -> (B x N x out_channels)
*/
Tensor* Conv1d::forward(Tensor* data) {
    this->input = data;

    size_t dim = data->getDim();
    DimVector shape_o = data->getShape();

    assert(shape_o[shape_o.size()-1] == in_channels);
    shape_o[shape_o.size()-1] = out_channels;

    size_t dim_sq = 1;
    for(size_t i=0; i<dim-1; i++) {
        dim_sq *= shape_o[i];
    }
    DimVector shape_forward = {dim_sq, in_channels};
    this->input->reshape(shape_forward);

    // (B*N x in_channel) @ (in_channel x out_channel)
    Tensor* mul = this->input->matmul(this->weight);

    if(this->bias) {
        this->output = mul->add(this->bias);
        delete mul;
    } else {
        this->output = mul;
    }

    this->output->reshape(shape_o);

    return output;
}

Tensor* Conv1d::backward(Tensor* gradients) {
    return nullptr;
}
