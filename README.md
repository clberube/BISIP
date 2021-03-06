# BISIP | Bayesian inversion of SIP data
[![Documentation Status](https://readthedocs.org/projects/bisip/badge/?version=latest)](https://bisip.readthedocs.io/en/latest/?badge=latest)
[![Build Status](https://www.travis-ci.com/clberube/BISIP.svg?branch=main)](https://www.travis-ci.com/clberube/BISIP)

BISIP is a fast, robust and open-source Python program for bayesian inversion of spectral induced polarization (SIP) data. It allows the evaluation of SIP parameters and their uncertainties by propagating measurement errors through the inversion process. Dias, Cole-Cole, Debye and Warburg decomposition schemes are currently implemented. Additional models will be included in the future.

In 2019 BISIP was re-written from the ground up with a powerful ensemble MCMC sampler, better code practice and improved [documentation](https://bisip.readthedocs.io/en/latest/). See our original [2017 paper](https://doi.org/10.1016/j.cageo.2017.05.001) in Computers & Geosciences.

<p align="center">
  <img src="/figures/ExampleFit_K389369.png" width="50%">
</p>

## Documentation
Visit https://bisip.readthedocs.io for the full documentation including API docs, tutorials and data file templates.

## Requirements
- [Python 3.6+](https://www.python.org/downloads/)
- [emcee](https://emcee.readthedocs.io/en/stable/)
- [NumPy](https://numpy.org/)
- [matplotlib](https://matplotlib.org/)

### Optional dependencies
- [tqdm](https://tqdm.github.io/): enables progress bars
- [corner](https://corner.readthedocs.io/en/latest/): enables beautiful corner plots

## Installation
Clone this repository to your computer. Then navigate to the bisip directory. Finally run the setup.py script with Python. The `-f` option forces a reinstall if a previous version of the package was already installed.

```zsh
$ git clone https://github.com/clberube/bisip
$ cd bisip
$ python setup.py install -f
```

## Quickstart
Using BISIP is as simple as importing a SIP model, initializing it and fitting it to a data set. BISIP also offers many utility functions for plotting and analyzing inversion results. See the [tutorials](https://bisip.readthedocs.io/en/latest/tutorials/quickstart.html) for more examples.

```python
from bisip import PolynomialDecomposition

# Define a data file to invert
filepath = '/path/to/DataFile_1.csv'
# Initialize the inversion model
model = PolynomialDecomposition(filepath=filepath,
                                nwalkers=32,  # number of walkers
                                nsteps=1000,  # number of MCMC steps
                                )
# Fit the model to this data file
model.fit()
```
```
Out:
100%|██████████| 1000/1000 [00:01<00:00, 558.64it/s]
```

## 2017 archive
The original BISIP code from the 2017 paper is archived in the `bisip1-archive` branch.
