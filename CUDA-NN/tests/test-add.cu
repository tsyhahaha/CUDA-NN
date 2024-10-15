void test_add() {
    const char* path = "/home/taosiyuan/cudaCode/CUDA-NN/data/bn1.weight.txt";
    float* h_o;

    DimVector shape1 = {3};
    float* h_d1 = loadWeightsFromTxt(path, {1,8});

    DimVector shape2 = {1, 3, 4};
    float* h_d2 = loadWeightsFromTxt(path, {8});

    printf("mat1:\n");
    printM(h_d1, shape1);

    printf("mat2:\n");
    printM(h_d2, shape2);

    Tensor* mat1 = new Tensor(h_d1, shape1);
    Tensor* mat2 = new Tensor(h_d2, shape2);

    mat1->unsqueeze(-1);
    printShape(mat1->getShape());

    mat1->scale_(2);
    printf("mat1 = 2 * mat1:\n");
    h_o = mat1->toHost();
    printM(h_o, shape1);
    free(h_o);

    Tensor* c = mat1->dot(mat2);
    h_o = c->toHost();
    printf("mat1 * mat2:\n");
    printM(h_o, c->getShape());

    c = mat1->sub(mat2);
    h_o = c->toHost();
    printf("mat1 - mat2:\n");
    printM(h_o, c->getShape());

    mat1->squeeze();
    c = mat1->add(mat2);
    h_o = c->toHost();
    printf("Broadcast: mat1(squeezed) + mat2:\n");
    printM(h_o, c->getShape());
}