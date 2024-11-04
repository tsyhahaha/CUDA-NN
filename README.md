# CUDA-NN
A collection of learning resources, featuring a straightforward implementation in Python and a comprehensive implementation of a neural network in CUDA.

* Basic CUDA code for learners to get started.
* A simple Python implementation of [PointNet](https://arxiv.org/abs/1612.00593).
* CUDA implementations of tensors, layers, and the PointNet model, including training and inference code.

## Basic CUDA Code
This section includes three main components:
* **Matmul**: The base version of matrix multiplication along with several optimized variations.
* **Reduce**: Different versions for summation operations on matrices.
* **Transpose**: Basic and optimized versions of matrix transpose.

Optimizations focus on the following aspects:
* **Shared Memory**: Tiling techniques and optimizations to minimize discrete reads.
* **Register Memory**: Increasing the ratio of computation to memory access to enhance FLOPS.
* **Bank Conflict**: Strategies to avoid bank conflicts caused by shared memory.
* **Memory Access Acceleration**: Using float4 reading to improve performance.

To run the starter code:
```bash
cd CUDA-NN/base/Matmul
nvcc v0.cu -o v0
./v0
```

### Helpful References:
* [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomic-functions)
* General-Purpose Graphics Processor Architecture
* [Optimizing a CUDA Matmul Kernel for cuBLAS-like Performance](https://siboehm.com/articles/22/CUDA-MMM)
* [NVIDIA_SGEMM_PRACTICE (GitHub)](https://github.com/wangzyon/NVIDIA_SGEMM_PRACTICE)
* [CUDA Matrix Multiplication: Implementation, Optimization, and Performance Analysis](https://chiemon.github.io/2020/02/06/CUDA-%E7%9F%A9%E9%98%B5%E4%B9%98%E6%B3%95-%E4%BC%98%E5%8C%96%E5%8F%8A%E6%80%A7%E8%83%BD%E5%88%86%E6%9E%90-%E4%B8%8A.html)
* [Ultimate Guide to CUDA Matrix Multiplication Optimization](https://zhuanlan.zhihu.com/p/410278370)
* [CUDA Matrix Transpose Optimization](https://code.hitori.moe/post/cuda-transpose-optimization/)
* [How to Optimize Algorithms in CUDA (GitHub)](https://github.com/BBuf/how-to-optim-algorithm-in-cuda)
* [CUDA Practice: Matrix Multiplication](http://www.zh0ngtian.tech/posts/975c867a.html)

## Simple Implementation of PointNet in Python
This section contains the code for the model implementation and training of PointNet on a single GPU.

To run:
```bash
cd CUDA-NN/python
python train.py
```

## Implementation of PointNet in CUDA
As a significant assignment for the UCAS GPU course, we were tasked with implementing PointNet inference and training in CUDA, achieving certain performance benchmarks in both accuracy and speed. This project serves as an excellent opportunity to practice CUDA.

The tensors, layers, and sub-modules of PointNet are implemented in `CUDA-NN/src`. Configure the data path in `CUDA-NN/src/test.cu` and the model parameters path in `CUDA-NN/src/CMakeLists.txt`.

To run the project:
```bash
cd CUDA-NN/CUDA-NN
mkdir build
cd build
cmake ..
make run/test
```

* `make run`: Reads test data from `CUDA-NN/data/beat` and saves the output to `CUDA-NN/data/cuout`.
* `make test`: Runs the official test program. Please ensure `hdf5` is installed and download the `h5` files for the MNIST dataset. This will read the test dataset and output both the running time and accuracy.

### Helpful Reference:
* [CUDA-DNN-MNIST (GitHub)](https://github.com/jpowie01/CUDA-DNN-MNIST)

## Additional Resources for the UCAS GPU Course
To avoid redundant wheel-building, this repository offers a basic framework for model inference, along with numerous suggestions and reusable tools. This repository achieves an inference speed of around 5 seconds, but there are still many areas for optimization due to limited effort. I hope this repository can help future students have more resources to explore deeper optimization techniques.

### 1. Profiling the Program
There are two primary tools available for profiling the program to analyze performance:
* **NVIDIA Nsight Systems (nsys)**
* **NVIDIA Nsight Compute (ncu)**

Be mindful of how to use these tools. Optimizing inference encompasses various aspects, so do not overlook any critical area:
* **Data Transmission**: `nsys` profiles the time consumed by many APIs like `cudaFree`, `cudaMalloc`, and `cudaMemcpy`, which can significantly impact inference time.
* **Kernel Performance**: `ncu` provides detailed insights into each running kernel, including execution time, memory reads/writes, resource usage, and optimization suggestions.

### Helpful Reference:
* [Profiling and Performance Analysis](https://blog.csdn.net/qq_44108731/article/details/140502836)

### 2. Merging the Project
Versions on the course server (Ubuntu 18.04):
* nvcc: 12.1
* gcc: 7.5

The profiler only supports submitting a single `.cu` file, so it’s necessary to merge the entire project into a single file. To run the merge script:

```bash
bash merge.sh test/main # Merge test.cu or main.cu
```

Then use the official `nvcc` to compile the `merged.cu` file as written in `compile.sh`:
```bash
nvcc merged.cu -o train -Xcompiler "-O3 -std=c++14" -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_62,code=sm_62 -gencode arch=compute_70,code=sm_70 -lhdf5_cpp -lhdf5

./train
```

*Note: The compilation process may fail due to the version of gcc or the architecture of the GPU. Please verify compatibility if errors occur.*

### 3. Verifying Program Accuracy
To validate the correctness of layers or models, it’s essential to compare them with the official implementation in Torch at the input/output level. To do this:

* Ensure that the absolute or relative paths used in both CUDA and Python programs are correct. It is recommended to use absolute paths instead of relative ones.
* Verify that the information in `config.yaml` matches.
* Ensure that `make run` executes successfully in the `CUDA-NN/build` directory.

Then run:
```bash
python beat.py
```

If all test points pass, you will see output like this:
```
Test (Python) module POINTNET
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:00<00:00, 21.80it/s]
--------------------------------------------------
Test (CUDA) module POINTNET
--------------------------------------------------
Outputs matched
--------------------------------------------------
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:00<00:00, 2445.66it/s]
[AC] All 10 tests passed successfully!
```