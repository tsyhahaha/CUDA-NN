#ifndef DATALOADER_H
#define DATALOADER_H

#include <vector>
#include <algorithm>
#include "tensor.cuh"

class DataLoader {
    private:
        size_t idx;
        size_t batchsize;
        size_t num_workers;
        bool pin_memory;
        bool drop_last;

        size_t cropping_size;
        size_t channel;

        std::vector<std::vector<float>> data;
        std::vector<int> labels;

    public:
        DataLoader(std::vector<std::vector<float>>& data, std::vector<int>& labels, size_t batchsize=1, size_t cropping_size=20000, size_t channel=3, size_t num_workers=0, bool pin_memory=false, bool drop_last=false);

        Tensor* getBatchedData(std::vector<int>& labels);

        size_t getDataNum();

        size_t getBatchNum();
    private:
        std::vector<float> crop(std::vector<float>& points);
};

#endif /* !DATALOADER_H */