'''
The Cython build script.  Generally don't call this directly -- use Makefile
instead.
'''


from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy as np

ext = Extension("scfbutils", sources=["scfbutils.pyx", "scfbutils_c.c"])

setup(name="scfbutils", ext_modules=cythonize([ext]), include_dirs=[np.get_include()])
