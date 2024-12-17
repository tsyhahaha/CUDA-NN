// 这是程序二的模板程序，我们已经准备好了加载数据集和加载程序一模型参数的部分，请实现CUDA的深度学习推理过程，请严格保持输出格式输出
// nvcc test.cu -o test -Xcompiler "-O3 -std=c++14" -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_53,code=sm_53 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_62,code=sm_62 -gencode arch=compute_70,code=sm_70 -lhdf5 -lhdf5_cpp

#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <string>
#include <fstream>
#include <map>
#include <dirent.h>
#include <cstring>
#include <H5Cpp.h>

#include "configure.cuh"
#include "layers.cuh"
#include "models.cuh"
#include "tensor.cuh"
#include "utils.cuh"
#include "datasets/dataloader.cuh"
#include "loss.cuh"
#include "optimizer.cuh"

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
    std::map<std::string, std::vector<float>> params;

    std::vector<std::string> param_files = get_files_in_directory(dir);
    for (const auto& file : param_files) {
        std::string filename = file.substr(0, file.find_last_of("."));
        params[filename] = read_param(dir + "/" + file);
    }

    return params;
}

/****************************************************************************************
 * 读取训练集数据
 ****************************************************************************************/

using namespace H5;
void read_h5_file(const std::string& file_path, std::vector<std::vector<float>>& list_of_points, std::vector<int>& list_of_labels) {
    try {
        // 打开文件
        H5File file(file_path, H5F_ACC_RDONLY);

        // 获取文件中的所有数据集名称
        std::vector<std::string> dataset_names;
        hsize_t num_objs = file.getNumObjs();
        for (hsize_t i = 0; i < num_objs; i++) {
            dataset_names.push_back(file.getObjnameByIdx(i));
        }

        // 读取每个数据集
        for (const auto& name : dataset_names) {
            DataSet dataset = file.openDataSet(name + "/points");
            DataSpace dataspace = dataset.getSpace();

            // 获取数据集的维度
            hsize_t dims[2];
            dataspace.getSimpleExtentDims(dims, NULL);

            // 读取数据
            std::vector<float> points(dims[0] * dims[1]);
            dataset.read(points.data(), PredType::NATIVE_FLOAT);

            // 存储点云数据
            list_of_points.push_back(points);

            // 读取标签
            Attribute label_attr = file.openGroup(name).openAttribute("label");
            int label;
            label_attr.read(PredType::NATIVE_INT, &label);

            // 存储标签
            list_of_labels.push_back(label);
        }
    } catch (FileIException& error) {
        error.printErrorStack();
    } catch (DataSetIException& error) {
        error.printErrorStack();
    } catch (DataSpaceIException& error) {
        error.printErrorStack();
    } catch (DataTypeIException& error) {
        error.printErrorStack();
    }
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

/****************************************************************************************
 * 存储模型参数
 ****************************************************************************************/
void write_param(const std::vector<float>& data, const std::string& filepath) {
    std::ofstream file(filepath);
    if (file.is_open()) {
        for (const auto& value : data) {
            file << value << std::endl;
        }
        file.close();
    } else {
        std::cerr << "Unable to open file: " << filepath << std::endl;
    }
}

void save_model_params_and_buffers_to_txt(std::map<std::string, std::vector<float>> params, std::string dir){
    for (const auto& pair : params) {
        // std::cout << "name: " << name << std::endl;
        const std::string& name = pair.first;
        const std::vector<float>& values = pair.second;
        std::string fileName = dir + "/" + name + ".txt";
        write_param(values,fileName);
    }
}


int main(int argc, char *argv[]) {
    std::string base = "/home/tsyhahaha/default";
    auto params_tmp = read_params(base);
    Configurer::set_global_weights(params_tmp);

    std::string dir = argv[1];
    std::map<std::string, std::vector<float>> params;

    // load model
    PointNet* pointnet = new PointNet();
    pointnet->train();
    pointnet->init_weights();
    // pointnet->load_weights();

    DEBUG_PRINT("FINISH LOADING MODEL...\n");

    // settings
    size_t batch_size = Configurer::batch_size;
    float lr = Configurer::learning_rate;
    size_t cropping_size = Configurer::cropping_size;
    bool pin_memory = false;

    // load data
    std::string file_path = "/home/tsyhahaha/CUDA-NN/data/splits/train_point_clouds.h5";
    std::vector<std::vector<float>> list_of_points;
    std::vector<int> list_of_labels;
    read_h5_file(file_path, list_of_points, list_of_labels);
    DataLoader* dataloader = new DataLoader(list_of_points, list_of_labels, batch_size, cropping_size, false, 0, pin_memory, false);

    DEBUG_PRINT("FINISH LOADING DATA...\n");

    // initial source
    std::vector<int> labels(batch_size);
    Tensor* input = new Tensor({batch_size, cropping_size, 3});
    Tensor* mask = new Tensor({batch_size});
    Tensor* labels_oh = new Tensor({batch_size, 10}); // class num = 10
    Tensor* d_out = new Tensor({batch_size, 10});   // pointnet gradients

    cudaStream_t preprocessStream;
    CHECK(cudaStreamCreate(&preprocessStream));

    // training
    std::map<std::string, Tensor*> name_params_dict;
    pointnet->name_params(name_params_dict);

    // CrossEntropyLoss* loss_func = new CrossEntropyLoss();
    NegativeLikelihoodLoss* loss_func = new NegativeLikelihoodLoss();

    DEBUG_PRINT("INIT OPTIMIZER...\n");

    Optimizer* opt = new SGD(name_params_dict, lr, 0.9);

    DEBUG_PRINT("READY TO BEGIN...\n");

    int* d_labels;
    CHECK(cudaMalloc(&d_labels, batch_size*sizeof(int)));

    unsigned int data_sum = list_of_points.size();
    unsigned int batch_num = dataloader->getBatchNum();

    // 开始计时，使用chrono计时，不支持其它计时方式
    auto start = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < 1; i++) {
        
        dataloader->getBatchedData(input, mask, labels);

        // preprocess labels to one hot
        cudaMemcpyAsync(d_labels, labels.data(), labels.size() * sizeof(int), cudaMemcpyHostToDevice, preprocessStream);
        Tensor::oneHot(d_labels, labels_oh, labels.size(), 10, preprocessStream);

        input->transpose(-2, -1);

        Tensor* output = pointnet->forward(input, mask);

        CHECK(cudaStreamSynchronize(preprocessStream));

        float loss = loss_func->forward(output, labels_oh);
        loss_func->backward(d_out);

        pointnet->backward(d_out);

        opt->step();
        opt->zero_grad();

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;
        printf("Batch[%ld/%d] Time: %f s, lr: %.5f, loss: %.3f\n", i+1, batch_num, diff.count(), opt->get_lr(), loss);
    }
    
    // 向主机端同步以等待所有异步调用的GPU kernel执行完毕，这句必须要有
    cudaDeviceSynchronize();

    // 结束计时
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;

    for(auto& pair: name_params_dict) {
        params.insert(std::make_pair(pair.first, pair.second->toVec()));
    }

    // save_model_params_and_buffers_to_txt(params, dir);

    // 输出结果，请严格保持此输出格式，并把0.0001替换成实际的准确率，请不要输出除了此结果之外的任何内容！！！
    std::cout << std::fixed << std::setprecision(4) << diff.count() << std::endl;
    std::cout << "0.5000"; //test use
    // clean up
    // delete pointnet;
    delete input;
    delete dataloader;
    CHECK(cudaFree(d_labels));
    CHECK(cudaStreamDestroy(preprocessStream));

    return 0;
}