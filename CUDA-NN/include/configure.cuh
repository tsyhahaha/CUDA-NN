#pragma once
#ifndef CONFIGURE_H
#define CONFIGURE_H

#define BLOCK_SIZE1D 256
#define BLOCK_SIZE2D 16
// less than the minimum value of the column and column of the matrix involved in the calculation
#define TILE_SIZE 16
#define BLOCK_SIZE3D 16

#define BATCH_BASE 4

#define TM 4
#define TN 4

#define DEFAULT_LEARNING_RATE 1e-5
#define DEFAULT_EPOCHS 100
#define DEFAULT_BATCH_SIZE 4
#define DEFAULT_CROPPING_SIZE 20000

#include<vector>
#include<string>
#include<map>
#include<stdio.h>

class Configurer {
private:
    static size_t getUIntValue(std::string variableName, size_t defaultValue);
    static int getIntValue(std::string variableName, int defaultValue);
    static float getFloatValue(std::string variableName, float defaultValue);
    static std::string getStringValue(std::string variableName, std::string defaultValue);
    static std::map<std::string, std::vector<float>> global_weight_map;

public:
    static float learning_rate;
    static size_t epochs;
    static size_t batch_size;
    static size_t cropping_size;
    static bool hook;

    // CUDA sizes
    static std::vector<float>& getWeights(std::string name);
    static void printCUDAConfiguration();
    static void set_global_weights(const std::map<std::string, std::vector<float>>& weight_map);

};

#endif /* !CONFIGURE_H */
