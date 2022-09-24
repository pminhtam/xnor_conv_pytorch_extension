from torch.utils.cpp_extension import load
binary_cpp = load(name="binary_cpp", sources=["binary.cpp"], verbose=True)
help(binary_cpp)
