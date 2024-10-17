#ifndef DATALOADER_H
#define DATALOADER_H

#include<vector>
#include "tensor.cuh"

class DataLoader {
    private:
        size_t idx = 0;
        size_t batchsize;
        size_t num_workers;
        bool pin_memory;
        bool drop_last;

        std::vector<std::vector<float>>& data;
        std::vector<int>& labels;

    public:
        DataLoader(std::vector<std::vector<float>>& data, std::vector<int>& labels, size_t batchsize=1, size_t num_workers=0, bool pin_memory=false, bool drop_last=false);

        Tensor* getBatchedData();

        size_t getDataNum();

        size_t getBatchNum();
};

#endif /* !DATALOADER_H */