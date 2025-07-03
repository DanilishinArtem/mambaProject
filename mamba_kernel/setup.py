from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='mamba_cuda',
    ext_modules=[
        CUDAExtension(
            name='mamba_cuda',
            sources=['mamba_scan.cpp', 'mamba_full_kernel.cu'],
        ),
    ],
    cmdclass={'build_ext': BuildExtension}
)
