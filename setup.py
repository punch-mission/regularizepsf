import numpy as np
from Cython.Build import cythonize
from setuptools import setup

setup(
    ext_modules=cythonize("regularizepsf/helper.pyx"),
    include_dirs=[np.get_include()]
)
