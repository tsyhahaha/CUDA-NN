#include "dataloader.cuh"

DataLoader::DataLoader(std::vector<std::vector<float>>& data, std::vector<int>& labels, size_t batchsize, size_t num_workers, bool pin_memory, bool drop_last) {
    this->batchsize = batchsize;
    this->num_workers = num_workers;
    this->pin_memory = pin_memory;
    this->drop_last = drop_last;

    this->data = data;
    this->labels = labels;
}

Tensor* DataLoader::getBatchedData() {
    return nullptr;
}

Tensor* DataLoader::getDataNum(){
    return this->labels.size();
}

Tensor* DataLoader::getBatchNum() {
    return drop_last ? (this->getDataNum()-1)/batchsize : (this->getDataNum()-1)/batchsize + 1;
}