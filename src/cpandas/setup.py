from distutils.core import setup
from Cython.Build import cythonize

#  python setup.py build_ext --inplace

setup(
    ext_modules=cythonize(
        "edge_calc.pyx", compiler_directives={"language_level": "3"}
    )
)