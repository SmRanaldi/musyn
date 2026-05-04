"""
Build script that compiles the optional Cython C extension for the adaptive
envelope hot loop. If Cython or NumPy headers are unavailable, the package
installs without the extension and falls back to Numba JIT or NumPy.
"""
from setuptools import setup

try:
    from Cython.Build import cythonize
    import numpy as np

    ext_modules = cythonize(
        "src/musyn/envelope/_loop.pyx",
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
            "nonecheck": False,
        },
        annotate=False,
    )
    include_dirs = [np.get_include()]
except (ImportError, Exception):
    ext_modules = []
    include_dirs = []

setup(ext_modules=ext_modules, include_dirs=include_dirs)
