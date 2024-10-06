#include "layers/linear.cuh"
#include "tensor/tensor.cuh"
#include "common/utils.cuh"

void test_linear() {
    size_t in_features = 3, out_features = 3;
    Linear* nn = new Linear(in_features, out_features, false, IDENTITY);

    size_t B = 8;
    DimVector shape = {B, 3};
    Tensor* x = new Tensor(shape);
    x->initialize(RANDOM);

    float* h_x = x->toHost();
    printf("h_x(%ldx%ld):\n", shape[0], shape[1]);
    print_M(h_x, shape);

    Tensor* out = nn->forward(x);

    float* h_o = out->toHost();
    printf("Linear(%ldx%ld) @ input(%ldx%ld)\n", in_features, out_features, shape[0], shape[1]);
    print_M(h_o, out->getShape());

}



int main() {
    test_linear();
    // test_mul();
}