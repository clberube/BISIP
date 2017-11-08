# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 13:25:48 2015

@author: Charles
"""

try:
    from setuptools import setup
    from setuptools import Extension
except ImportError:
    from distutils.core import setup
    from distutils.extension import Extension

from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize("bisip/cython_funcs.pyx"),
    include_dirs=[numpy.get_include()]
)