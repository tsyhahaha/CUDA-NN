# Analysis of the Model structure

In our 3D MINST recognition task, PointNet is the base model to be implemented in CUDA. 

See related code in `CUDA-NN/python/model.py`. 

## Overview

<img src="../asserts/pointnet.png">

In this proj, the blue classification net is what we concentrate on. Overall, the model is constructed by 2 parts including:
* Feature Encoder: extract feature from point cloud by point alignment, pooling etc.
* Classification Head: a couple of Linears.

## Feature Encoder

Consisting 3 steps:
* predict the transform matrix(3x3) to align the points (B N 3)
* After raising dimension by Conv1d (B N 64), predict the transform matrix (64x64) once more to align the feature.
* Raising dimension to (B N 1024) and max pooling to (B 1024).

## Classificaton Head

```
# x is the output of Feature Encoder
x = F.relu(self.bn1(self.fc1(x)))
x = F.relu(self.bn2(self.dropout(self.fc2(x))))
x = self.fc3(x)
x = F.log_softmax(x, dim=1)
```

## Conclusion

Matrix operations involved:
* Transpose
* Batch Matmul
* Reduce(max, mean)

Layer involved:
* Linear
* Conv1d
* BatchNorm
* MaxPool
* ReLU
* Sigmoid
* Dropout

