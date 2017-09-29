# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 16:18:50 2017

@author: Charles
"""

from setuptools import setup, find_packages

from distutils.core import setup
setup(
  name = 'BISIP',
  packages=find_packages(), # this must be the same as the name above
  version = '0.1',
  license = 'MIT',
  install_requires=['pymc'],
  description = 'Bayesian inversion of SIP data',
  long_description = 'README.md',
  author = 'Charles L. Berube',
  author_email = 'cberube@ageophysics.com',
  url = 'https://github.com/clberube/BISIP', # use the URL to the github repo
  download_url = 'https://github.com/clberube/BISIP/archive/0.1.tar.gz', # I'll explain this in a second
  keywords = ['testing', 'logging', 'example'], # arbitrary keywords
  classifiers = [],
)
