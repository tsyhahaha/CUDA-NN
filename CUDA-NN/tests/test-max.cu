void test_max() {
    const char* path = "/home/taosiyuan/cudaCode/CUDA-NN/data/bn1.weight.txt";
    float* h_o;

    DimVector shape = {2, 3, 4};
    float* h_d = loadWeightsFromTxt(path, shape);

    printf("mat:\n");
    printM(h_d, shape);

    Tensor* mat = new Tensor(h_d, shape);

    printf("begin to max\n");

    mat->max_(-1, false);

    printf("finished max\n");

    h_o = mat->toHost();

    printM(h_o, mat->getShape());
}