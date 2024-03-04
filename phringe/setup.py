import numpy as np
from Cython.Build import cythonize
from setuptools import setup

setup(
    ext_modules=cythonize(r'phringe/core/processing/_complex_amplitude.pyx'),
    include_dirs=[np.get_include()]
)
