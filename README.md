# C++/CUDA Extensions XNOR convolution in PyTorch

XNOR extension 

In XNOR convolution, both the filters and the input to convolutional layers are binary. Now, by approximating the convolution operations with XNOR and bitcounting operations, we can gain massive speed-up and memory savings.

xnor convolution using bitwise operation 

Implement on numpy-python, Cpp and CUDA. 


- Inspect the C++ and CUDA extensions in the `cpp/` and `cuda/` folders,

# Build cpp and CUDA 

Build C++ and/or CUDA extensions by going into the `cpp/` or `cuda/` folder and executing `python setup.py install`,

### Cpp

```shell
cd cpp
python setup.py install 
```

### CUDA
```shell
cd cuda
python setup.py install 
```

# Use 
### Cpp 
```python
import binary_cpp
output = binary_cpp.binary_conv2d(input,filter,bias)
```

### CUDA 
```python
import binary_cuda
output = binary_cuda.binary_conv2d_cuda(input,filter,bias)
```

### Numpy

```python
from py.xnor_bitwise_numpy import xnor_bitwise_np
out = xnor_bitwise_np(input,filter)
```
# References

[1] https://github.com/pytorch/extension-cpp

[2] Rastegari, Mohammad, Vicente Ordonez, Joseph Redmon, and Ali Farhadi. "Xnor-net: Imagenet classification using binary convolutional neural networks." In European conference on computer vision, pp. 525-542. Springer, Cham, 2016.

[3] https://github.com/cooooorn/Pytorch-XNOR-Net

[4] https://github.com/anilsathyan7/ConvAcc

[5][Binarized Neural Networks: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1](https://arxiv.org/pdf/1602.02830.pdf)

