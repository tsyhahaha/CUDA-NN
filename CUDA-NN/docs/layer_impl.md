# Layer Implementation Details

Include implementation of layers like: Linear, Conv1d. For the implementation of tensors, refer to `CUDA-NN/cuda/tensor`.

## Tensor

Tensor is the minimal operand in DNN. At the very beginning, we have to figure out the attributes and methods of tensors.

### Attributes
* `float* dataPtr`: the source data
* `Device device`: use enum class to rep device(CPU or GPU)
* `size_t dim`: the dimension such as 1(D), 2(D), 3(D)
* `size_t* sizes`: the length equals to dim.

### methods
* Constructors and destructors
* `size_t getDim()`: get the dimension of the tensor
* `size_t getSize(size_t dim)`: get the size of specific dim
* Set value: initialize, from CPU
* get value: if needed
* Unary operation: `transpose`, `max`...
* Binary operation: `add`, `multiply`
* utils: `print`

## Linear
