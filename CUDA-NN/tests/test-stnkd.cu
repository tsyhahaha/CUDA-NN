void test_stnkd() {
    const char* path = "/home/taosiyuan/cudaCode/CUDA-NN/data/bn1.weight.txt";
    float* h_o;

    size_t k = 4;

    DimVector shape = {2, k, 32};
    float* h_d = loadWeightsFromTxt(path, shape);

    Tensor* input = new Tensor(h_d, shape);

    printf("input:\n");
    printM(h_d, shape);

    STNkd* nn = new STNkd(k);

    Tensor* o = nn->forward(input);

    h_o = o->toHost();
    printf("output:\n");
    printM(h_o, o->getShape());
}
