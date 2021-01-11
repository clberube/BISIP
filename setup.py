#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: cberube
# @Date:   05-03-2020
# @Email:  charles.lafreniere-berube@polymtl.ca
# @Last modified by:   charles
# @Last modified time: 2020-03-12T22:10:53-04:00
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 16:18:50 2017

@author: Charles
"""

from setuptools import setup, find_packages
from distutils.extension import Extension
try:
    import numpy
except ImportError:
    from setuptools import dist
    dist.Distribution().fetch_build_eggs(['numpy'])
    import numpy


SRC_DIR = 'src'
PACKAGES = find_packages(where=SRC_DIR)
PREREQ = ['setuptools>=18.0', 'cython', 'numpy']

thelibFolder = os.path.dirname(os.path.realpath(__file__))
requirementPath = thelibFolder + '/requirements'
REQUIRES = []  # To populate by reading the requirements file
if os.path.isfile(requirementPath):
    with open(requirementPath) as f:
        REQUIRES = f.read().splitlines()

cmdclass = {}
EXT_MODULES = [Extension("bisip.cython_funcs",
                         sources=["src/bisip/cython_funcs.pyx"])]

setup(
    name='bisip',
    setup_requires=PREREQ,
    packages=PACKAGES,
    package_dir={"": SRC_DIR},
    version='1.1.1',
    license='MIT',
    install_requires=REQUIRES,
    description='Bayesian inversion of SIP data',
    long_description='README.md',
    author='Charles L. Bérubé',
    author_email='charles.lafreniere-berube@polymtl.ca',
    url='https://github.com/clberube/bisip',
    keywords=['stochastic inversion', 'spectral induced polarization', 'mcmc'],
    classifiers=[],
    cmdclass=cmdclass,
    ext_modules=EXT_MODULES,
    include_dirs=[numpy.get_include()],
    include_package_data=True,
)
