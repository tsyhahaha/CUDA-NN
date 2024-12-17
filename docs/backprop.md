# Backpropgation 

The derivation of the backpropagation for certain layers in the implementation. 

Definition:

| Name                            | Notation |
| ------------------------------- | -------- |
| loss                            | $L$      |
| output                          | $y$      |
| $\frac{\partial L}{\partial y}$ | $d_y$    |

## Layers

Including linear, con1d, bn1d.

### 1.Linear

| NAME         | SHAPE               |
| ------------ | ------------------- |
| input: $x$   | $(N, C_{in})$       |
| weights: $W$ | $(C_{out}, C_{in})$ |
| bias: $b$    | $(C_{out})$         |
| output: $y$  | $(N, C_{out})$      |

***forward***
$$
y = x\times W^T+b
$$
***backward***
$$
\frac{\partial L}{\partial W} =\frac{\partial L}{\partial y}\cdot \frac{\partial y}{\partial W}=x^T \times d_y\\
shape:(C_{out},N)\times (N, C_{in})\rightarrow(C_{out}, C_{in})
$$

$$
\frac{\partial L}{\partial b} = \frac{\partial L}{\partial y}\cdot \frac{\partial y}{\partial b}=d_y\\
shape: (N, C_{out})\xrightarrow{\Sigma} (C_{out})
$$

> gradients on bias need to sum at other dimensions to match the shape.

$$
\frac{\partial L}{\partial x}=\frac{\partial L}{\partial y}\cdot \frac{\partial y}{\partial x}=d_y\times W \\
shape：(N,C_{out})\times (C_{out}, C_{in})\rightarrow (N, C_{in})
$$

### 2.BatchNorm1d

| NAME         | SHAPE                  |
| ------------ | ---------------------- |
| input: $x$   | $(N,C)$ or $(N, C, L)$ |
| weights: $W$ | $(C)$                  |
| bias: $b$    | $(C)$                  |
| output: $y$  | $(N, C)$ or $(N,C,L)$  |

***forward***
$$
\hat{x}=\frac{x-\mu[x]}{\sqrt{var[x]+eps}}\overset{notes}{=}\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}} \\
y=\hat{x}\cdot W+b
$$

> $W,b$ here are also noted as $\gamma, \beta$.

***backward***

Firstly, the gradients w.t. learnable params are:
$$
\frac{\partial L}{\partial W}=\sum_{i=1}^Nd_{y_i}\cdot \hat{x_i}\\
shape:(N,C)/(N,C,L)\cdot (N,C)/(N,C,L)\xrightarrow{\Sigma}(C)
$$

$$
\frac{\partial L}{\partial b}=\sum_{i=1}^Nd_{y_i}\\
shape: (N,C)/(N,C,L)\xrightarrow{\Sigma}(C)
$$

> Here is the inner product, not matrix multiplication.

Secondly, we have to compute the gradients w.t. the input of this layer, 
$$
\frac{\partial L}{\partial x}=d_y\cdot \frac{\partial y}{\partial x}
$$

You can refer to the computational graph in this [page](https://www.adityaagrawal.net/blog/deep_learning/bprop_batch_norm). According to the derivative chain rule of multivariate function and the graph, 
$$
\begin{aligned}
\frac{\partial L}{\partial x_i}=\frac{\partial L}{\partial \hat x_i}\frac{\partial \hat x_i}{\partial x_i}+\frac{\partial L}{\partial \mu}\frac{\partial \mu}{\partial x_i} + \frac{\partial L}{\partial \sigma^2}\frac{\partial \sigma^2}{\partial x_i}
\end{aligned}
$$
Before getting the final result, we need to figure out several sub-formula(sub-computational-graph).
$$
\frac{\partial L}{\partial \hat x_i}=\frac{\partial L}{\partial y_i}\frac{\partial y_i}{\partial\hat x_i}=d_{y_i}\cdot W \\
shape:(C)/(C,L)\cdot (C)\rightarrow (C)/(C,L)
$$

$$
\begin{aligned}
\frac{\partial L}{\partial \sigma^2}&=\sum_{i=1}^N \frac{\partial L}{\partial \hat x_i}\frac{\partial\hat x_i}{\partial \sigma^2}\\
&=\sum_{i=1}^N \frac{\partial L}{\partial \hat x_i}\cdot\frac{-(x_i-\mu)}{2(\sigma^2+\epsilon)^{3/2}} \\
\frac{\partial L}{\partial \mu}&=\sum_{i=1}^N\frac{\partial L}{\partial \hat x_i}\frac{\partial \hat x_i}{\partial \mu} + \frac{\partial L}{\partial \sigma^2}\frac{\partial \sigma^2}{\partial \mu}\\
&=\sum_{i=1}^N \frac{\partial L}{\partial \hat x_i}\frac{-1}{\sqrt{\sigma^2+\epsilon}} + \frac{\partial L}{\partial \sigma^2}\frac{\sum_{i=1}^N -2(x_i-\mu)}{N}\\
&=\sum_{i=1}^N \frac{\partial L}{\partial \hat x_i}\frac{-1}{\sqrt{\sigma^2+\epsilon}} 
\end{aligned}
$$

And evidently,

| Notations                                                    | Shape         |
| ------------------------------------------------------------ | ------------- |
| $\frac{\partial \hat x_i}{\partial x_i}=\frac{1}{\sqrt{\sigma^2+\epsilon}}$ | $(C)/(C,L)$   |
| $\frac{\partial \sigma^2}{\partial x_i}=\frac{2(x_i-\mu)}{N}$ | $(C)/(C,L)$   |
| $\frac{\partial \mu}{\partial x_i}=\frac{1}{N}$              | Scalar: $(1)$ |

So the final result,
$$
\frac{\partial L}{\partial x_i}=\frac{\partial L}{\partial \hat x_i}\frac{1}{\sqrt{\sigma^2+\epsilon}}+\frac{\partial L}{\partial \sigma^2}\frac{2(x_i-\mu)}{N}+\frac{\partial L}{\partial \mu}\frac{1}{N}
$$

> *Cached data should be reused as much as possible.*
> Cached data: $\frac{1}{\sqrt{\sigma^2+\epsilon}}, (x-\mu),\hat x$ 

> Real implementation merge the ReLU into the BN1d, so the real gradients need to be filtered positions where output is non-positive(abs).

### 3.Conv1d

| NAME         | SHAPE               |
| ------------ | ------------------- |
| input: $x$   | $(N,C_{in},L)$      |
| weights: $W$ | $(C_{out}, C_{in})$ |
| bias: $b$    | $(C_{out})$         |
| output: $y$  | $(N,C_{out},L)$     |

***forward***
$$
y = \mathrm{BatchMatMul}(x, W^T)+b
$$

> The implementation here has been simplified, because kernel size equals 1 in the NNs.

***backward***

Firstly, gradients w.t. learnable params,
$$
\frac{\partial L}{\partial W_{j,k}}=\sum_{i=1}^N[\mathrm{BatchMatMul}(\frac{\partial L}{\partial y}, x^T)]_{i,j,k}\\
shape:(N,C_{out},L)\times(N,L,C_{in})\xrightarrow{\Sigma} (C_{out},C_{in})\\
\frac{\partial L}{\partial b_j}=\sum_{i=1,k=1}^{N,L} [\frac{\partial L}{\partial y}]_{i,j,k}\\
shape:(N,C_{out},L)\xrightarrow{\Sigma}(C_{out})
$$
Secondly, gradients w.t. input,
$$
\frac{\partial L}{\partial x}=\mathrm{BatchMatMul}(\frac{\partial L}{\partial y}, W^T)\\
shape: (N,C_{out},L)\times (C_{in},C_{out})\rightarrow (N, C_{in}, L)
$$



## *Ops*

Some other operations related to the backprop. Mainly include: max, bmm...

### 1.max

| NAME        | SHAPE     |
| ----------- | --------- |
| input: $x$  | $(N,C,L)$ |
| output: $y$ | $(N,C)$   |

***forward***
$$
y=x.max(-1)
$$
assuming that the max index is $s$, a vector of shape $(N,C)$, then

***backward***
$$
\frac{\partial L}{\partial x}=\frac{\partial L}{\partial y}=d_y[...,:]\\
shape:(N,C)\rightarrow (N,C,L)
$$

### 2.bmm

| NAME        | SHAPE       |
| ----------- | ----------- |
| input: $x$  | $(N,C)$     |
| trans: $t$  | $(N, C, C)$ |
| output: $y$ | $(N,C)$     |

***forward***
$$
y=\mathrm{BatchMatMul}(x, t)\\
(N,L,C)\times (N,C,C)\rightarrow (N,L,C)
$$
***backward***
$$
\frac{\partial L}{\partial x}=\frac{\partial L}{\partial y}\frac{\partial y}{\partial x}=\mathrm{BatchMatMul}(d_y, t)\\
shape: (N,L,C)\times (N,C,C)\xrightarrow{bmm} (N, L, C)
$$

$$
\frac{\partial L}{\partial t}=\frac{\partial L}{\partial y}\frac{\partial y}{\partial t}=\mathrm{BatchMatMul}(d_y,x)\\
shape: (N,C,L)\times(N,L,C)\xrightarrow{bmm} (N,C,C)
$$

