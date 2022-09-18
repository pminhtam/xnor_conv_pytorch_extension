from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
# import os
# from distutils.sysconfig import get_config_vars
# (opt,) = get_config_vars('OPT')
# os.environ['OPT'] = " ".join(
#     flag for flag in opt.split() if flag != '-Wstrict-prototypes'
# )

# headers = ['libpopcnt.h','matmul.h']
# ,'binop_cuda_kernel.cu'
setup(
    name='binary_cuda',
    version='0.1',
    author = 'pminhtam',
    description = 'xnor cuda',
    ext_modules=[
        CUDAExtension(name='binary_cuda',
                     sources= ['binary_cuda.cpp']),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
