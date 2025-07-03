from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='mamba_scan',
    ext_modules=[
        CUDAExtension(
            name='mamba_scan',
            sources=['mamba_scan.cpp', 'mamba_scan_kernel.cu'],
        ),
    ],
    cmdclass={'build_ext': BuildExtension}
)
