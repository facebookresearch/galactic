from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

setup(
    name="bps_pytorch",
    ext_modules=[
        CUDAExtension(
            name="bps_pytorch",
            sources=["pytorch.cpp"],
            extra_compile_args=[],
            extra_link_args=[],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
