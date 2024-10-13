#include <iostream>
#include <vector>
#include <iomanip>
#include <string>
#include <fstream>
#include <map>
#include <dirent.h>
#include <cstring>

#include "layers.cuh"
#include "models.cuh"
#include "tensor.cuh"
#include "utils.cuh"
#include "configure.cuh"

// 获取目录中的所有 .txt 文件
std::vector<std::string> get_files_in_directory(const std::string& dir) {
    std::vector<std::string> files;
    DIR* dp;
    struct dirent* entry;
    if ((dp = opendir(dir.c_str())) != NULL) {
        while ((entry = readdir(dp)) != NULL) {
            std::string filename = entry->d_name;
            if (filename.find(".txt") != std::string::npos) {
                files.push_back(filename);
            }
        }
        closedir(dp);
    } else {
        perror("opendir");
    }
    return files;
}

// 读取 .txt 文件并转换为 std::vector<float>
std::vector<float> read_param(const std::string& filepath) {
    std::vector<float> data;
    std::ifstream file(filepath);
    if (file.is_open()) {
        float value;
        while (file >> value) {
            data.push_back(value);
        }
        file.close();
    } else {
        std::cerr << "Unable to open file: " << filepath << std::endl;
    }
    return data;
}

std::map<std::string, std::vector<float>> read_params(std::string dir) {
    // std::string dir = "."; // 当前目录
    std::map<std::string, std::vector<float>> params;

    // 获取目录中的所有 .txt 文件
    std::vector<std::string> param_files = get_files_in_directory(dir);
    for (const auto& file : param_files) {
        std::string filename = file.substr(0, file.find_last_of(".")); // 获取不带扩展名的文件名
        params[filename] = read_param(dir + "/" + file);
    }

    // // 访问参数时可以使用 params["conv1_weight"]
    // for (const auto& kv : params) {
    //     std::cout << "Key: " << kv.first << ", Values: ";
    //     // for (const auto& value : kv.second) {
    //     //     std::cout << value << " ";
    //     // }
    //     std::cout << std::endl;
    // }

    return params;
}

void test_params_load() {
    std::string dir = "/home/course/taosiyuan241/best_model";
    
    // 读取模型参数
    auto params = read_params(dir);
    Configurer::set_global_weights(params);

    PointNet* nn = new PointNet();

    nn->load_weights();

    printf("load model done\n");

    const char* path = "/home/course/taosiyuan241/CUDA-NN/data/bn1.weight.txt";
    float* h_o;

    DimVector shape = {2, 3, 4};
    float* h_d1 = loadWeightsFromTxt(path, shape);

    Tensor* input = new Tensor(h_d1, shape);

    Tensor* o = nn->forward(input);

    h_o = o->toHost();

    printM(h_o, o->getShape());

}

int main() {
    test_params_load();
}