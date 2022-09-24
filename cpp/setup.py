from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension
# import os
# from distutils.sysconfig import get_config_vars
# (opt,) = get_config_vars('OPT')
# os.environ['OPT'] = " ".join(
#     flag for flag in opt.split() if flag != '-Wstrict-prototypes'
# )

include_dirs = ['libpopcnt.h','binary_kernel.h']
setup(
    name='binary_cpp',
    version='0.1',
    author = 'pminhtam',
    description = 'xnor cpp',
    ext_modules=[
        CppExtension(name='binary_cpp',
                     sources= ['binary.cpp']),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
