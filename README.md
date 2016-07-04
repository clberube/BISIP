# BISIP
## Bayesian inference of spectral induced polarization parameters (Python 2.7)

**1. Installation**

  Clone or Download repository to local folder

  Install Python dependencies (Python 2.7, NumPy, matplotlib, SciPy, PyMC, Tkinter)
  
**2. Usage**
  
  Run BISIP_GUI.py to start the GUI
  
  OR

  Import the inversion function using:
  
  `from BISIP_models import mcmcSIPinv`
  
  And obtain results using all default arguments
  
  `sol = mcmcSIPinv("ColeCole", "/Documents/DataFiles/DATA.dat")`
  
  The full list of optional arguments is:
  
  `sol = mcmcSIPinv(model="ColeCole", filename="/Documents/DataFiles/DATA.dat", mcmc=mcmc_params, headers=1, ph_units="mrad", cc_modes=2, debye_poly=4, keep_traces=False)`
  
  Where `mcmc_params` is a python dictionary:
  `mcmc_params = {"nb_chain"   : 1,
               "nb_iter"    : 50000,
               "nb_burn"    : 40000,
               "thin"       : 1,
               "tune_inter" : 1000,
               "prop_scale" : 1.0,
               "verbose"    : 0,
                }`
  
  (See run_BISIP.py for an example script on how to use the function)

This code uses part of the PyMC package (https://github.com/pymc-devs/pymc)

