#ifndef DATALOADER_H
#define DATALOADER_H

#include <vector>
#include <algorithm>
#include <random>
#include <memory>
#include <stdexcept>

#include "tensor.cuh"

class DataLoader {
    private:
        size_t idx;
        size_t batchsize;
        size_t num_workers;
        bool pin_memory;
        bool drop_last;
        bool shuffle;

        size_t cropping_size;
        size_t channel;
        cudaStream_t stream;

        std::vector<std::vector<float>> data;
        std::vector<int> labels;

        float* prefetched_data = nullptr;
        float* prefetched_mask = nullptr;
        float* pinned_data = nullptr;

    public:
        DataLoader(std::vector<std::vector<float>>& data, std::vector<int>& labels, size_t batchsize=1, size_t cropping_size=30000, bool shuffle=false, size_t num_workers=0, bool pin_memory=false, bool drop_last=false);
        ~DataLoader();

        void getBatchedData(Tensor* &batched_input, Tensor*& batched_mask, std::vector<int>& labels);

        size_t getSize();

        size_t getBatchNum();
    private:
        std::vector<float> crop(std::vector<float>& points);
        void loadNextBatchAsync();
};

#endif /* !DATALOADER_H */