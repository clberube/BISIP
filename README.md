# BISIP | Bayesian inversion of SIP data
[![Documentation Status](https://readthedocs.org/projects/bisip/badge/?version=latest)](https://bisip.readthedocs.io/en/latest/?badge=latest)

BISIP is a fast, robust and open-source inversion program for laboratory spectral induced polarization (SIP) data. It allows propagation of data uncertainty onto parameters of various empirical SIP models. Additional mechanistic models will be included in the future.

BISIP was re-written from the ground up with a powerful ensemble MCMC sampler, better coding practice and improved documentation in 2019. See our original 2017 paper in [Computers & Geosciences](https://doi.org/10.1016/j.cageo.2017.05.001).

<img src="/figures/ExampleFit_K389369.png" width="40%" class="center">

## Requirements
- Python 3 (BISIP is developed on Python 3.8)  
- emcee (https://emcee.readthedocs.io/en/stable/)
- numpy (https://numpy.org/)
- matplotlib (https://matplotlib.org/)

## Documentation
Visit https://bisip.readthedocs.io/en/latest/ to consult the full documentation including API docs, tutorials and examples.

## Installation
Clone this repository to your computer. Then navigate to the bisip directory. Finally run the setup.py script with Python. The `-f` option forces a reinstall if the package is already present.

```
git clone https://github.com/clberube/bisip
cd bisip2
python setup.py install -f
```

## Quickstart
Using BISIP is as simple as importing a SIP model, initializing it and fitting it to a data set.
BISIP also offers many utility functions for plotting and analyzing inversion results. See the [tutorials](https://bisip.readthedocs.io/en/latest/tutorials/quickstart.html)
for more examples.

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

## Original version
The original BISIP code from 2017 is archived in the `bisip1-archive` branch.
