# C++/CUDA Extensions binary convolution in PyTorch

XNOR extension 


There are a few "sights" you can metaphorically visit in this repository:

- Inspect the C++ and CUDA extensions in the `cpp/` and `cuda/` folders,
- Build C++ and/or CUDA extensions by going into the `cpp/` or `cuda/` folder and executing `python setup.py install`,
- JIT-compile C++ and/or CUDA extensions by going into the `cpp/` or `cuda/` folder and calling `python jit.py`, which will JIT-compile the extension and load it,
- Benchmark Python vs. C++ vs. CUDA by running `python benchmark.py {py, cpp, cuda} [--cuda]`,
- Run gradient checks on the code by running `python grad_check.py {py, cpp, cuda} [--cuda]`.
- Run output checks on the code by running `python check.py {forward, backward} [--cuda]`.
