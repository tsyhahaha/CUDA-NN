#include "layers.cuh"
#include "tensor.cuh"
#include "utils.cuh"

void test_conv1d() {
    const char* path = "/home/taosiyuan/cudaCode/CUDA-NN/data/bn1.weight.txt";
    float* h_o;

    size_t in_channels = 3, out_channels = 8;
    size_t B = 2, N = 2;

    // weight
    DimVector shape1 = {out_channels, in_channels};
    float* h_d1 = loadWeights(path, shape1[0], shape1[1]);

    // bias
    DimVector shape2 = {out_channels};
    float* h_d2 = loadWeights(path, shape2[0], 1);

    Conv1d* nn = new Conv1d(in_channels, out_channels, 1);

    nn->load_weights(h_d1, h_d2, shape1, shape2);

    DimVector shape_in = {B, B, in_channels};
    Tensor* input = new Tensor(shape_in, RANDOM);

    Tensor* o = nn->forward(input);
    h_o = o->toHost();

    printf("Conv1d->forward:\n");
    print_M(h_o, o->getShape());
}


int main() {
    test_conv1d();
}