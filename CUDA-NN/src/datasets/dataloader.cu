#include "dataloader.cuh"

float* allocatePinnedMemory(size_t size) {
    float* ptr;
    cudaMallocHost(&ptr, size * sizeof(float));
    return ptr;
}

void freePinnedMemory(float* ptr) {
    cudaFreeHost(ptr);
}

std::vector<float> binaryAssign(int cropping_size, int L) {
    std::vector<float> vec(cropping_size);
    for (size_t i = 0; i < vec.size(); ++i) {
        if (i < L) {
            vec[i] = 1.0f;
        } else {
            vec[i] = 0.0f;
        }
    }
    return vec;
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

    CHECK(cudaStreamCreate(&stream));
    this->data = data;
    this->labels = labels;

    loadNextBatchAsync();

    if(pin_memory)
        CHECK(cudaHostAlloc((void**)&pinned_data, batchsize * 3 * cropping_size*sizeof(float), cudaHostAllocDefault));

    if(shuffle) {
        std::random_device rd;
        std::mt19937 g(rd());

        std::shuffle(data.begin(), data.end(), g);
    }

}

DataLoader::~DataLoader() {
    if (pin_memory) {
        cudaFreeHost(pinned_data);
    }
    cudaStreamDestroy(stream);
}

void DataLoader::loadNextBatchAsync() {
    if(this->prefetched_data == nullptr) {
        size_t nBytes = batchsize * cropping_size * 3 * sizeof(float);
        CHECK(cudaMalloc((float**)&prefetched_data, nBytes));
    }

    if(this->prefetched_mask == nullptr) {
        size_t nBytes = batchsize * sizeof(float);
        CHECK(cudaMalloc((float**)&prefetched_mask, nBytes));
    }
    std::vector<float> batch;
    std::vector<float> mask;
    DimVector shape;

    int idx = this->idx;
    if (idx < data.size()) {
        if (idx + batchsize > data.size() && !drop_last) {
            shape = {this->data.size() - idx, cropping_size, 3};
            batch.reserve((this->data.size() - idx) * cropping_size * 3);

            for (; idx < this->data.size(); idx++) {
                mask.push_back(this->data[idx].size()/3);
                std::vector<float> point = crop(this->data[idx]);
                batch.insert(batch.end(), point.begin(), point.end());
            }
        } else {
            if (idx + batchsize > this->labels.size()) {
                throw std::out_of_range("Index out of range for labels");
            }
            shape = {batchsize, cropping_size, 3};
            batch.reserve(batchsize * cropping_size * 3);

            for (int i = 0; i < batchsize; i++) {
                if (idx >= this->data.size()) {
                    throw std::out_of_range("Index out of range for data");
                }
                mask.push_back(this->data[this->idx].size()/3);
                std::vector<float> point = crop(this->data[idx]);
                batch.insert(batch.end(), point.begin(), point.end());
                idx++;
            }
        }
    } else {
        ERROR("No more data to load: %ld >= %ld\n", this->idx, this->data.size());
    }
    size_t size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
    CHECK(cudaMemcpyAsync(prefetched_data, batch.data(), size*sizeof(float), cudaMemcpyHostToDevice, stream));
    CHECK(cudaMemcpyAsync(prefetched_mask, mask.data(), shape[0]*sizeof(float), cudaMemcpyHostToDevice, stream));
}

std::vector<float> DataLoader::crop(std::vector<float>& points) {
    size_t new_size = std::min(points.size(), this->cropping_size * 3);
    
    points.resize(new_size);
    
    points.resize(this->cropping_size * 3, 0.0f);
    return points;
}

void DataLoader::getBatchedData(Tensor* &batched_input, Tensor* &batched_mask, std::vector<int>& labels) {
    DimVector shape;
    if (this->pin_memory) {
        // it seems to be slowly if needed to mv data from CPU to pin mem?
    } else {
        if (this->idx < data.size()) {
            if (this->idx + batchsize > data.size() && !drop_last) {
                labels.assign(this->labels.begin() + this->idx, this->labels.end());

                shape = {this->data.size() - this->idx, cropping_size, 3};
                this->idx = data.size();
            } else {
                if (this->idx + batchsize > this->labels.size()) {
                    throw std::out_of_range("Index out of range for labels");
                }
                labels.assign(this->labels.begin() + this->idx, this->labels.begin() + this->idx + batchsize);
                shape = {batchsize, cropping_size, 3};
                this->idx += batchsize;
            }
        } else {
            throw std::runtime_error("No more data to load");
        }

        size_t size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
        if(batched_input != nullptr && batched_input->getSize() < size) {
            delete batched_input;
            batched_input = nullptr;
        }
        if(batched_input == nullptr) {
            batched_input = new Tensor(shape);
            printShape(batched_input->getShape());
            DEBUG_PRINT("batched input is newly alloced!\n");
        }

        cudaStreamSynchronize(stream);
        float* tmp = batched_input->getData();
        batched_input->setData(prefetched_data);
        batched_input->setShape(shape);
        this->prefetched_data = tmp;

        tmp = batched_mask->getData();
        batched_mask->setData(prefetched_mask);
        batched_mask->setShape({shape[0]});
        this->prefetched_mask = tmp;

        if(this->idx < data.size())
            loadNextBatchAsync();
    }
}

size_t DataLoader::getSize(){
    return (size_t)this->labels.size();
}

size_t DataLoader::getBatchNum() {
    return (size_t)(!drop_last ? (this->getSize()-1) / batchsize + 1:this->getSize() / batchsize);
}