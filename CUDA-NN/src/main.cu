#include "utils.cuh"
#include "tensor.cuh"

void test_add() {
    const char* path = "/home/taosiyuan/cudaCode/CUDA-NN/data/bn1.weight.txt";
    float* h_o;
    size_t M = 1, N = 8;
    float* h_d = loadWeights(path, M, N);

    size_t M2 = 2, N2 = 8;
    float* h_d2 = loadWeights(path, M2, N2);

    DimVector shape = {M, N};
    printf("mat1(%ldx%ld):\n", M, N);
    print_M(h_d, shape);

    DimVector shape2 = {M2, N2};
    printf("mat2(%ldx%ld):\n", M2, N2);
    print_M(h_d2, shape2);

    Tensor* a = new Tensor(h_d, shape);
    Tensor* a2 = new Tensor(h_d2, shape2);

    a->scale_(2);
    printf("mat1 = 2 * mat1:\n");
    print_M(a->toHost(), a->getShape());

    Tensor* c = a->add(a2);
    h_o = c->toHost();
    printf("mat1 + mat2:\n");
    print_M(h_o, c->getShape());

    c = a->sub(a2);
    h_o = c->toHost();
    printf("mat1 - mat2:\n");
    print_M(h_o, c->getShape());

    a->squeeze();
    c = a->add(a2);
    h_o = c->toHost();
    printf("mat1(squeezed) + mat2:\n");
    print_M(h_o, c->getShape());


}

void hostMatmul(
    float *h_a, float *h_b, float *h_result, int m, int n, int k
) {
    float tmp = 0.0f;
    for (int i = 0; i < m; ++i) 
    {
        for (int j = 0; j < k; ++j) 
        {
            for (int h = 0; h < n; ++h) 
            {
                tmp += h_a[i * n + h] * h_b[h * k + j];
            }
            h_result[i * k + j] = tmp;
            tmp = 0.0f;
        }
    }
}

void test_mul() {
    const char* path = "/home/taosiyuan/cudaCode/CUDA-NN/data/bn1.weight.txt";
    float* h_o;
    size_t M = 2, N = 3;
    float* h_d = loadWeights(path, M, N);

    size_t M2 = 3, N2 = 3;
    float* h_d2 = loadWeights(path, M2, N2);

    DimVector shape = {M, N};
    printf("mat1(%ldx%ld):\n", M, N);
    print_M(h_d, shape);

    DimVector shape2 = {M2, N2};
    printf("mat2(%ldx%ld):\n", M2, N2);
    print_M(h_d2, shape2);

    Tensor* a = new Tensor(h_d, shape);
    Tensor* a2 = new Tensor(h_d2, shape2);

    a2->transpose();
    h_o = a2->toHost();
    // printf("Transpose(a):\n");
    // print_M(h_o, a2->getShape());

    // CPU version
    float* h_result = (float*)malloc(M*M2*sizeof(float));
    hostMatmul(h_d, h_o, h_result, M, N, M2);
    printf("host matmul:\n");
    print_M(h_result, {M, M2});

    // a2->transpose();
    // Tensor* o = a->matmul(a2);
    // float* out = o->toHost();
    // print_M(out, o->getShape());

    Tensor* tmp = new Tensor(shape2);
    tmp->initialize(IDENTITY);
    float* out = tmp->toHost();
    print_M(out, tmp->getShape());

    Tensor* o = a->matmul(tmp);
    out = o->toHost();
    print_M(out, o->getShape());


    delete o, a, a2;
}

int main() {
    // test_add();
    test_mul();
}