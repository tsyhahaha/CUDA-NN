#include "layers.cuh"
#include "models.cuh"
#include "tensor.cuh"
#include "utils.cuh"

void test_pointnet() {
    const char* path = "/home/taosiyuan/cudaCode/CUDA-NN/data/bn1.weight.txt";
    float* h_o;

    DimVector shape = {2, 3, 4};
    float* h_d1 = loadWeightsFromTxt(path, shape);

    Tensor* input = new Tensor(h_d1, shape);

    PointNet* pointnet = new PointNet();
    Tensor* o = pointnet->forward(input);

    h_o = o->toHost();

    printM(h_o, o->getShape());
}

int main() {
    test_pointnet();
}