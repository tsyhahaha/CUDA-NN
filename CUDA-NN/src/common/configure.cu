#include "configure.cuh"
#include "utils.cuh"


void Configurer::set_global_weights(const std::map<std::string, std::vector<float>>& weight_map) {
    // for (const auto& pair : weight_map) {
    //     DEBUG_PRINT("%s\n", pair.first.c_str());
    // }
    global_weight_map = weight_map;
}

std::vector<float>& Configurer::getWeights(std::string name) {
     if (global_weight_map.find(name) == global_weight_map.end()) {
        throw std::runtime_error("Error: Weight parameter '" + name + "' does not exist.");
    }
    
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

bool Configurer::getBoolValue(std::string variableName, bool defaultValue) {
    char* envValue = std::getenv(variableName.c_str());
    if (!envValue) {
        return defaultValue;
    }
    std::string envStr(envValue);
    if (envStr == "true" || envStr == "1" || envStr == "yes") {
        return true;
    }
    return false;
}


size_t Configurer::getUIntValue(std::string variableName, size_t defaultValue) {
        char* envValue = std::getenv(variableName.c_str());
        if (!envValue) {
            return defaultValue;
        }
        return static_cast<size_t>(std::strtoull(envValue, nullptr, 10));
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

float Configurer::learning_rate = getFloatValue("LEARNING_RATE", DEFAULT_LEARNING_RATE);
size_t Configurer::epochs = getUIntValue("EPOCHS", DEFAULT_EPOCHS);
size_t Configurer::batch_size = getUIntValue("BATCH_SIZE", DEFAULT_BATCH_SIZE);
size_t Configurer::cropping_size = getUIntValue("CROPPING_SIZE", DEFAULT_CROPPING_SIZE);
bool Configurer::hook = getBoolValue("HOOK_ENABLED", DEFAULT_HOOK_ENABLED
);
bool Configurer::track_grads = getBoolValue("TRACK_GRADS", DEFAULT_TRACK_GRADS);
std::string Configurer::target = getStringValue("TARGET", DEFAULT_TARGET);
