#include "configure.cuh"
#include "utils.cuh"


void Configurer::set_global_weights(const std::map<std::string, std::vector<float>>& weight_map) {
    for (const auto& pair : weight_map) {
        DEBUG_PRINT("%s\n", pair.first.c_str());
    }
    global_weight_map = weight_map;
}

std::vector<float>& Configurer::getWeights(std::string name) {
    DEBUG_PRINT("loading %s(%ld)\n", name.c_str(), global_weight_map[name].size());
    return global_weight_map[name];
}

int Configurer::getIntValue(std::string variableName, int defaultValue) {
    char* envValue = std::getenv(variableName.c_str());
    if (!envValue) {
        return defaultValue;
    }
    return std::atoi(envValue);
}

float Configurer::getFloatValue(std::string variableName, float defaultValue) {
    char* envValue = std::getenv(variableName.c_str());
    if (!envValue) {
        return defaultValue;
    }
    return atof(envValue);
}

std::string Configurer::getStringValue(std::string variableName, std::string defaultValue) {
    char* envValue = std::getenv(variableName.c_str());
    if (!envValue) {
        return defaultValue;
    }
    return std::string(envValue);
}

void Configurer::printCUDAConfiguration() {
    cudaDeviceProp properties;
    cudaGetDeviceProperties(&properties, 0);  // Takes only one (first) GPU

    printf("=====================================\n");
    printf("         CUDA Configurer\n");
    printf("=====================================\n");
    printf(" Device name: %s\n", properties.name);
    printf(" Memory Clock Rate (KHz): %d\n", properties.memoryClockRate);
    printf(" Memory Bus Width (bits): %d\n", properties.memoryBusWidth);
    printf("-------------------------------------\n");
    printf("\n");
}

std::map<std::string, std::vector<float>> Configurer::global_weight_map;

// float Configurer::learningRate = getFloatValue("LEARNING_RATE", DEFAULT_LEARNING_RATE);
// int Configurer::numberOfEpochs = getIntValue("NUMBER_OF_EPOCHS", DEFAULT_NUMBER_OF_EPOCHS);
// int Configurer::batchSize = getIntValue("BATCH_SIZE", DEFAULT_BATCH_SIZE);

// int Configurer::tensorAddBlockSize = getIntValue("TENSOR2D_ADD_BLOCK_SIZE", DEFAULT_TENSOR_ADD_BLOCK_SIZE);
// int Configurer::tensorSubtractBlockSize = getIntValue("TENSOR2D_SUBTRACT_BLOCK_SIZE", DEFAULT_TENSOR2D_SUBTRACT_BLOCK_SIZE);
// int Configurer::tensorScaleBlockSize = getIntValue("TENSOR2D_SCALE_BLOCK_SIZE", DEFAULT_TENSOR2D_SCALE_BLOCK_SIZE);
// int Configurer::tensorMultiplyBlockSize = getIntValue("TENSOR2D_MULTIPLY_BLOCK_SIZE", DEFAULT_TENSOR2D_MULTIPLY_BLOCK_SIZE);
// int Configurer::tensorMultiplyBlockNumber = getIntValue("TENSOR2D_MULTIPLY_BLOCK_NUMBER", DEFAULT_TENSOR2D_MULTIPLY_BLOCK_NUMBER);
// int Configurer::tensorMultiplySharedMemory = getIntValue("TENSOR2D_MULTIPLY_SHARED_MEMORY", DEFAULT_TENSOR2D_MULTIPLY_SHARED_MEMORY);
// int Configurer::tensorMeanBlockSize = getIntValue("TENSOR2D_MEAN_BLOCK_SIZE", DEFAULT_TENSOR2D_MEAN_BLOCK_SIZE);

// int Configurer::reLuBlockSize = getIntValue("RELU_BLOCK_SIZE", DEFAULT_TENSOR2D_RELU_BLOCK_SIZE);