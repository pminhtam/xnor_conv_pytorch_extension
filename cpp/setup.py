from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension
# import os
# from distutils.sysconfig import get_config_vars
# (opt,) = get_config_vars('OPT')
# os.environ['OPT'] = " ".join(
#     flag for flag in opt.split() if flag != '-Wstrict-prototypes'
# )
from torch.__config__ import parallel_info
def parallel_backend():
    parallel_info_string = parallel_info()
    parallel_info_array = parallel_info_string.splitlines()
    backend_lines = [line for line in parallel_info_array if line.startswith('ATen parallel backend:')]
    if len(backend_lines) != 1:
        return None
    backend = backend_lines[0].rsplit(': ')[1]
    return backend


def CppParallelExtension(name, sources, *args, **kwargs):
    parallel_extra_compile_args = []

    backend = parallel_backend()

    if (backend == 'OpenMP'):
        parallel_extra_compile_args = ['-DAT_PARALLEL_OPENMP', '-fopenmp']
    elif (backend == 'native thread pool'):
        parallel_extra_compile_args = ['-DAT_PARALLEL_NATIVE']
    elif (backend == 'native thread pool and TBB'):
        parallel_extra_compile_args = ['-DAT_PARALLEL_NATIVE_TBB']

    extra_compile_args = kwargs.get('extra_compile_args', [])
    extra_compile_args += parallel_extra_compile_args
    kwargs['extra_compile_args'] = extra_compile_args

    return CppExtension(name, sources, *args, **kwargs)


include_dirs = ['libpopcnt.h','binary_kernel.h']
setup(
    name='binary_cpp',
    version='0.1',
    author = 'pminhtam',
    description = 'xnor cpp',
    ext_modules=[
        CppParallelExtension(name='binary_cpp',
                     sources= ['binary.cpp']),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
