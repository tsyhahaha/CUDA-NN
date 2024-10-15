void test_bmm() {
    const char* path = "/home/taosiyuan/cudaCode/CUDA-NN/data/bn1.weight.txt";
    float* h_o;

    DimVector shape1 = {2, 3, 4};
    float* h_d1 = loadWeightsFromTxt(path, shape1);

    DimVector shape2 = {2, 4, 3};
    float* h_d2 = loadWeightsFromTxt(path, shape2);

    printf("mat1:\n");
    printM(h_d1, shape1);

    printf("mat2:\n");
    printM(h_d2, shape2);

    Tensor* mat1 = new Tensor(h_d1, shape1);
    Tensor* mat2 = new Tensor(h_d2, shape2);
    
    Tensor* o = mat1->bmm(mat2);
    h_o = o->toHost();
    printM(h_o, o->getShape());
}