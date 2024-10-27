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
    // std::string dir = "."; // 当前目录
    std::map<std::string, std::vector<float>> params;

    // 获取目录中的所有 .txt 文件
    std::vector<std::string> param_files = get_files_in_directory(dir);
    for (const auto& file : param_files) {
        std::string filename = file.substr(0, file.find_last_of(".")); // 获取不带扩展名的文件名
        std::cout << filename << "\n";
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


// 范例kernel函数，无实际作用
__global__ void add_arrays(int* a, int* b, int* c, int size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size) {
        c[index] = a[index] + b[index];
    }
}

int main(int argc, char *argv[]) {
    
    std::string dir = argv[1];  // 第一个参数是程序所在的目录，这个目录是存放前一步训练模型参数文件的目录，从这个目录下读取模型参数文件，相对于这个目录读取测试集点云数据和标签
    // cout << dir;
    
    // 读取模型参数
    auto params = read_params(dir);
    
    Configurer::set_global_weights(params);
    PointNet* pointnet = new PointNet();
    pointnet->load_weights();

    std::string file_path = "/home/tsyhahaha/CUDA-NN/data/splits/test_point_clouds.h5";
    std::vector<std::vector<float>> list_of_points;
    std::vector<int> list_of_labels;
    // 读取训练集数据
    read_h5_file(file_path, list_of_points, list_of_labels);

    DataLoader* dataloader = new DataLoader(list_of_points, list_of_labels, 8, 10240, true);

    // 开始计时，使用chrono计时，不支持其它计时方式
    auto start = std::chrono::high_resolution_clock::now();

    unsigned int right_num = 0;
    unsigned int data_sum = list_of_points.size();
    unsigned int batch_num = dataloader->getBatchNum();
    
    for (size_t i = 0; i < 1; i++) {
        std::vector<int> labels;

        Tensor* input = dataloader->getBatchedData(labels);
        input->transpose(-2, -1);
        Tensor* output = pointnet->forward(input);
        Tensor* pred = output->argmax(-1);
        pred->squeeze();

        float* pred_labels = pred->toHost();

        printM(pred_labels, pred->getShape());
        for(int label: labels) {
            printf("%d ", label);
        }
        printf("\n");
        // clean up
        delete input, output, pred;

    
        // std::cout << "[" << i+1 << "/" << data_sum << "] Points " << i << ", size=" << n_data << ": ";
        // printf("%d", int(pred_label[0]+0.5));
        // // for (const auto& point : list_of_points[i]) {
        // //     std::cout << point << " ";
        // // }
        // std::cout << "\tLabel: " << list_of_labels[i] << std::endl;

        printf("Batch[%d/%d] ", (int)i+1, (int)batch_num);

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;
        std::cout << "Time: " << diff.count() << "s" << std::endl;
    }
    
    // 向主机端同步以等待所有异步调用的GPU kernel执行完毕，这句必须要有
    cudaDeviceSynchronize();

    // 结束计时
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;

    // 输出结果，请严格保持此输出格式，并把0.0001替换成实际的准确率，请不要输出除了此结果之外的任何内容！！！
    std::cout << std::fixed << std::setprecision(4) << diff.count() << right_num << "/" << list_of_points.size() << " (" << (right_num / (float)list_of_points.size()) * 100 << "%)" << std::endl;

    return 0;
}