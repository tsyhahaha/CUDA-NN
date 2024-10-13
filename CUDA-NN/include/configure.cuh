#pragma once
#ifndef CONF_H
#define CONF_H

#define BLOCK_SIZE1D 4
#define BLOCK_SIZE2D 4  
// less than the minimum value of the column and column of the matrix involved in the calculation
#define TILE_SIZE 2
#define BLOCK_SIZE3D 4  

#include<vector>
#include<map>
#include<stdio.h>

class Configurer {
private:
    static int getIntValue(std::string variableName, int defaultValue);
    static float getFloatValue(std::string variableName, float defaultValue);
    static std::string getStringValue(std::string variableName, std::string defaultValue);
    static std::map<std::string, std::vector<float>> global_weight_map;

public:
    // training
    static float learningRate;
    static int numberOfEpochs;
    static int batchSize;

    // CUDA sizes
    static int tensorAddBlockSize;
    static int tensorSubtractBlockSize;
    static int tensorScaleBlockSize;
    static int tensorMultiplyBlockSize;
    static int tensorMultiplyBlockNumber;
    static int tensorMultiplySharedMemory;
    static int tensorMeanBlockSize;

    static int reLuBlockSize;

    static std::vector<float>& getWeights(std::string name);
    static void printCUDAConfiguration();
    static void set_global_weights(const std::map<std::string, std::vector<float>>& weight_map);

};

#endif /* !CONF_H */