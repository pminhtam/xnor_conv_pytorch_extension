from torch.utils.cpp_extension import load
binary_cuda = load(
    'binary_cuda', ['binary_cuda.cpp','binop_cuda_kernel.cu'], verbose=True)

help(binary_cuda)
