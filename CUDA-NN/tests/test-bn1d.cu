void test_bn1d(){
    const char* path = "/home/taosiyuan/cudaCode/CUDA-NN/data/bn1.weight.txt";
    float* h_o;

    size_t C = 3;
    
    BatchNorm1d* bn1d = new BatchNorm1d(C);

    float* h_d1 = loadWeightsFromTxt(path, {C});
    float* h_d2 = loadWeightsFromTxt(path, {C});

    // bn1d->load_weights(h_d1, h_d2, {C}, {C});
    printM(h_d1, {C});
    printM(h_d2, {C});

    bn1d->load_weights(h_d1, {C}, "mean");
    bn1d->load_weights(h_d2, {C}, "var");

    // input (N, C) or (N, C, L)
    size_t N = 2, L = 4;

    Tensor* input = new Tensor({N, C, L}, RANDOM);
    printM(input->toHost(), input->getShape());

    Tensor* o = bn1d->forward(input);

    float* d_o = o->toHost();

    printM(d_o, o->getShape());
}