// 这是程序二的模板程序，我们已经准备好了加载数据集和加载程序一模型参数的部分，请实现CUDA的深度学习推理过程，请严格保持输出格式输出
// 编译的命令为：nvcc test.cu -o test -Xcompiler "-O3 -std=c++14" -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_53,code=sm_53 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_62,code=sm_62 -gencode arch=compute_70,code=sm_70 -lhdf5 -lhdf5_cpp

#include <vector>
#include <chrono>
#include <iomanip>
#include <cstring>
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <string>
#include <dirent.h>

#include <sys/stat.h>
#include <sys/types.h>

#include "configure.cuh"
#include "layers.cuh"
#include "models.cuh"
#include "tensor.cuh"
#include "utils.cuh"

#define toSizet(value) std::stoull(value)

// 定义配置结构
using ConfigMap = std::map<std::string, std::string>;  // 全局配置项
using LayerParams = std::map<std::string, std::map<std::string, std::string>>;  // 层的参数

// 解析 YAML 文件
std::pair<ConfigMap, LayerParams> loadYamlConfig(const std::string& filename) {
    std::ifstream file(filename);
    ConfigMap config;
    LayerParams layers;

    std::string line;
    std::string current_layer;
    bool inside_layer = false;

    while (std::getline(file, line)) {
        // 移除首尾空格
        line.erase(0, line.find_first_not_of(" \t"));
        line.erase(line.find_last_not_of(" \t") + 1);

        // 跳过空行或注释
        if (line.empty() || line[0] == '#') {
            continue;
        }

        // 查找键值对的冒号
        std::size_t colon_pos = line.find(':');
        if (colon_pos != std::string::npos) {
            std::string key = line.substr(0, colon_pos);
            std::string value = line.substr(colon_pos + 1);

            // 移除键和值的多余空格
            key.erase(0, key.find_first_not_of(" \t"));
            key.erase(key.find_last_not_of(" \t") + 1);
            value.erase(0, value.find_first_not_of(" \t\""));
            value.erase(value.find_last_not_of(" \t\"") + 1);

            if(value == "") {
                inside_layer = false;
            }

            if (!inside_layer) {
                // 如果当前是全局配置项
                if (value != "") {
                    config[key] = value;
                } else {
                    // 开始新 layer 的定义
                    current_layer = key;
                    inside_layer = true;
                }
            } else {
                // 如果在 layer 内部，解析 layer 的参数
                layers[current_layer][key] = value;
            }
        } else {
            // 当遇到新 section 时，结束当前 layer
            inside_layer = false;
        }
    }

    return {config, layers};
}


void printConfig(const ConfigMap& config, const std::map<std::string, std::string>& layer_params) {
    std::cout << "General Config:\n\n";
    for (ConfigMap::const_iterator it = config.begin(); it != config.end(); ++it) {
        std::cout << it->first << ": " << it->second << '\n';
    }

    std::cout << "\nLayer Params:\n\n";
    for (std::map<std::string, std::string>::const_iterator it = layer_params.begin(); it != layer_params.end(); ++it) {
        std::cout << it->first << ": " << it->second << '\n';
    }
}

bool create_directory(const std::string& path) {
    // 使用 mkdir 创建目录，0777 是目录的权限
    if (mkdir(path.c_str(), 0777) == 0) {
        std::cout << "Directory created: " << path << std::endl;
        return true;
    } else {
        if (errno == EEXIST) {
            DEBUG_PRINT("Directory already exists: %s\n", path.c_str());
            return true;  // 目录已经存在也认为是成功
        } else {
            ERROR("Error creating directory: %s", strerror(errno));
            return false;
        }
    }
}

/****************************************************************************************
 * 读取模型参数
 ****************************************************************************************/
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
        ERROR("Failed to read files in dir %s\n", dir.c_str());
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
        ERROR("Unable to open file: %s\n", filepath.c_str());
    }
    return data;
}

std::vector<size_t> read_shape(const std::string& filepath) {
    std::vector<size_t> data;
    std::ifstream file(filepath);
    if (file.is_open()) {
        float value;
        while (file >> value) {
            data.push_back(value);
        }
        file.close();
    } else {
        ERROR("Unable to open file: %s\n", filepath.c_str());
    }
    return data;
}

void save_vector_to_txt(const std::string& filename, const std::vector<float>& data) {
    std::ofstream file(filename);
    
    if (file) {
        for (const float& value : data) {
            file << value << "\n";
        }
        file.close();
    } else {
        ERROR("Unable to open file: %s\n", filename.c_str());
    }
}

std::map<std::string, std::vector<float>> read_params(std::string& dir) {
    std::map<std::string, std::vector<float>> params;

    std::vector<std::string> param_files = get_files_in_directory(dir);
    for (const auto& file : param_files) {
        std::string filename = file.substr(0, file.find_last_of(".")); // 获取不带扩展名的文件名
        params[filename] = read_param(dir + "/" + file);
    }

    return params;
}

std::string getBaseName(const std::string& filename) {
    size_t last_dot = filename.find_last_of('.');
    if (last_dot != std::string::npos) {
        return filename.substr(0, last_dot);
    }
    return filename; 
}

auto getModel(const std::string& target, const std::string& name, std::map<std::string, std::string>& cfg, std::string& param_file) {
    Module* nn;
    if(target=="model") {
        Configurer::set_global_weights(read_params(param_file));
        nn = new PointNet();
        static_cast<PointNet*>(nn)->load_weights();
        if(name == "encoder") {
            nn = static_cast<PointNet*>(nn)->feat;
        } else if(name == "stn3d") {
            nn = static_cast<PointNet*>(nn)->feat->stn;
        } else if(name == "stnkd") {
            nn = static_cast<PointNet*>(nn)->feat->fstn;
        }
        return nn;
    } else if(target == "layer") {
        // prepare to load the model to be tested
        const std::string weights_file = param_file + ".weight.txt";
        const std::string bias_file = param_file + ".bias.txt";
        auto weight_params = read_param(weights_file);
        auto bias_params = read_param(bias_file);

        if(name == "linear") {
            nn = new Linear(toSizet(cfg["in_features"]), toSizet(cfg["out_features"]));
            static_cast<Linear*>(nn)->load_weights(weight_params, "weights");
            static_cast<Linear*>(nn)->load_weights(bias_params, "bias");
        } else if(name=="batchnorm1d") {
            nn = new BatchNorm1d(toSizet(cfg["in_channels"]));
            static_cast<BatchNorm1d*>(nn)->load_weights(weight_params, "weights");
            static_cast<BatchNorm1d*>(nn)->load_weights(bias_params, "bias");
            auto running_mean = read_param(param_file + ".running_mean.txt");
            auto running_var = read_param(param_file + ".running_var.txt");
            static_cast<BatchNorm1d*>(nn)->load_weights(running_mean, "mean");
            static_cast<BatchNorm1d*>(nn)->load_weights(running_var, "var");
        } else if(name=="conv1d") {
            nn = new Conv1d(toSizet(cfg["in_channels"]), toSizet(cfg["out_channels"]), 1);
            static_cast<Conv1d*>(nn)->load_weights(weight_params, "weights");
            static_cast<Conv1d*>(nn)->load_weights(bias_params, "bias");
        } else {
            ERROR("Not implemented!\n");
        }
        return nn;
    } else {
        ERROR("Target: %s not implemented!\n", target.c_str());
    }
}

void test_module(
    std::string& target, std::string& name, 
    std::string& param_file, std::string& data_dir, 
    std::map<std::string, std::string> cfg){

    auto nn = getModel(target, name, cfg, param_file);
    DEBUG_PRINT("Load weights to %s\n", name.c_str());

    // load beat test points
    std::string test_dir = data_dir + "/beat";
    std::string output_dir = data_dir + "/cuout";
    create_directory(output_dir);
    std::vector<std::string> test_points;
    test_points = get_files_in_directory(test_dir);

    auto start = std::chrono::high_resolution_clock::now();
    for(std::string test_file: test_points) {
        if(test_file.find("shape") == std::string::npos) {
            std::string base_name = getBaseName(test_file);

            test_file = test_dir + "/" + base_name;
            DEBUG_PRINT("[TESTING] %s\n", test_file.c_str());

            std::vector<float> data_vec = read_param(test_file + ".txt");
            std::vector<size_t> shape = read_shape(test_file + ".shape.txt");

            Tensor* input = new Tensor(data_vec.data(), shape);
            Tensor* output = nn->forward(input);

            std::vector<float> output_vec = output->toVec();
            std::string output_file = output_dir + "/" + base_name + ".txt";
            save_vector_to_txt(output_file, output_vec);
        }
    }
    
    cudaDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;

    std::cout << std::fixed << std::setprecision(4) << diff.count() << "s\n";
}

void test_op(
    std::string& target, std::string& name, 
    std::string& param_file, std::string& data_dir, 
    std::map<std::string, std::string> cfg){

    // load beat test points
    std::string test_dir = data_dir + "/beat";
    std::string output_dir = data_dir + "/cuout";
    create_directory(output_dir);
    std::vector<std::string> test_points;
    test_points = get_files_in_directory(test_dir);

    auto start = std::chrono::high_resolution_clock::now();
    for(std::string test_file: test_points) {
        if(test_file.find("shape") == std::string::npos) {
            std::string base_name = getBaseName(test_file);
            test_file = test_dir + "/" + base_name;
            DEBUG_PRINT("[TESTING] %s\n", test_file.c_str());
            std::vector<float> data_vec = read_param(test_file + ".txt");
            std::vector<size_t> shape = read_shape(test_file + ".shape.txt");
            Tensor* input = new Tensor(data_vec.data(), shape);
            std::vector<float> output_vec;
            if(name == "max") {
                input->max_(2, false);
                output_vec = input->toVec();
            } else {
                ERROR("Not implemented op %s!\n", name.c_str());
            }

            std::string output_file = output_dir + "/" + base_name + ".txt";
            save_vector_to_txt(output_file, output_vec);
        }
    }
    
    cudaDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;

    std::cout << std::fixed << std::setprecision(4) << diff.count() << "s\n";
}

/* support to test single layer with its pretrained weights */
int main(int argc, char *argv[]) {

    std::string filename = "/home/taosiyuan/cudaCode/CUDA-NN/config.yaml";
    
    // Parse the yaml configuration
    std::pair<ConfigMap, LayerParams> config = loadYamlConfig(filename);
    std::string target = config.first["target"];
    std::string name = config.first["name"];
    if (target == "layer" && config.second.find(name) != config.second.end()) {
        printConfig(config.first, config.second[name]);
    } else if(target=="layer") {
        ERROR("Error: Layer %s not found\n", name.c_str());
    }

    std::string param_file = config.first["param_path"];   // weights filename without postfix like .weights.txt, .bias.txt
    std::string data_dir = config.first["data_dir"];     // this test points dir

    DEBUG_PRINT("Finished loading yaml config.\n");
    std::map<std::string, std::string> cfg = config.second[name];
    
    if(target=="model" || target=="layer") {
        test_module(target, name, param_file, data_dir, cfg);
    } else if(target == "op") {
        test_op(target, name, param_file, data_dir, cfg);
    } else {
        ERROR("Not implemented target %s\n", target.c_str());
    }
    

    return 0;
}
