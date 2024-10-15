void test_conv1d() {
    const char* path = "/home/taosiyuan/cudaCode/CUDA-NN/data/bn1.weight.txt";
    float* h_o;

    size_t in_channels = 3, out_channels = 5;
    size_t B = 2, N = 2;

    // weight
    DimVector shape1 = {out_channels, in_channels};
    float* h_d1 = loadWeightsFromTxt(path, shape1);

    // bias
    DimVector shape2 = {out_channels};
    float* h_d2 = loadWeightsFromTxt(path, shape2);

    Conv1d* nn = new Conv1d(in_channels, out_channels, 1);
    printf("begin to load\n");
    nn->load_weights(h_d1, h_d2, shape1, shape2);

    DimVector shape_in = {B, in_channels, N};
    Tensor* input = new Tensor(shape_in, RANDOM);
    printM(input->toHost(), input->getShape());

    Tensor* o = nn->forward(input);
    h_o = o->toHost();

    printf("Conv1d->forward:\n");
    printM(h_o, o->getShape());
}