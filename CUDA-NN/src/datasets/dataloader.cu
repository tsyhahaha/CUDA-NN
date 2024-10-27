#include "dataloader.cuh"

float* allocatePinnedMemory(size_t size) {
    float* ptr;
    cudaMallocHost(&ptr, size * sizeof(float));
    return ptr;
}

void freePinnedMemory(float* ptr) {
    cudaFreeHost(ptr);
}

DataLoader::DataLoader(std::vector<std::vector<float>>& data, std::vector<int>& labels, size_t batchsize, size_t cropping_size, bool shuffle, size_t num_workers, bool pin_memory, bool drop_last) {
    this->batchsize = batchsize;
    this->num_workers = num_workers;
    this->pin_memory = pin_memory;
    this->drop_last = drop_last;
    this->shuffle = shuffle;

    this->idx = 0;
    this->cropping_size = cropping_size; // cropping length, default: 21950
    // this->channel = channel;

    this->data = data;
    this->labels = labels;

    if(shuffle) {
        std::random_device rd;
        std::mt19937 g(rd());

        std::shuffle(data.begin(), data.end(), g);
    }
}

std::vector<float> DataLoader::crop(std::vector<float>& points) {
    std::vector<float> cropped_points(3 * cropping_size, 0.0f);
    size_t size = std::min(points.size(), this->cropping_size);
    std::copy(points.begin(), points.begin() + size, cropped_points.begin());
    return cropped_points;
}

Tensor* DataLoader::getBatchedData(std::vector<int>& labels) {
    DimVector shape;
    if(this->pin_memory) {
        // TODO
        return nullptr;
    } else {
        std::vector<float> batch;

        if(this->idx < data.size() && this->idx +batchsize >= data.size() && !drop_last) {
            shape = {this->data.size() - this->idx, cropping_size, 3};
            batch.reserve((this->data.size() - this->idx) * cropping_size * 3);

            for(; this->idx < this->data.size(); this->idx++) {
                std::vector<float> point = crop(this->data[this->idx]);
                batch.insert(batch.end(), point.begin(), point.end());
            }
            labels = std::vector<int>(this->labels.begin() + this->idx, this->labels.end());
        } else {
            shape = {batchsize, cropping_size, 3};
            batch.reserve(batchsize * cropping_size * 3);
            for(int i = 0; i<batchsize; i++) {
                std::vector<float> point = crop(this->data[this->idx]);
                batch.insert(batch.end(), point.begin(), point.end());
                this->idx++;
            }
            labels = std::vector<int>(this->labels.begin() + this->idx, this->labels.begin() + this->idx + batchsize);
        }

        Tensor* batch_input = new Tensor(shape);


        batch_input->fromVec(batch);
        printShape(batch_input->getShape());

        return batch_input;
    }
}

size_t DataLoader::getDataNum(){
    return (size_t)this->labels.size();
}

size_t DataLoader::getBatchNum() {
    return (size_t)(!drop_last ? (this->getDataNum()-1) / batchsize + 1:this->getDataNum() / batchsize);
}