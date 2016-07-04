# BISIP
## Bayesian inference of spectral induced polarization parameters (Python 2.7)

**1. Installation**

  Clone or Download repository to local folder

  Install Python dependencies (Python 2.7, NumPy, matplotlib, SciPy, PyMC, Tkinter)
  
**2. Usage**
  
  Run BISIP_GUI.py to start the GUI
  
  **_OR_**

  Import the inversion function using:
  
  `from BISIP_models import mcmcSIPinv`
  
  And obtain results using all default arguments and MCMC parameters with:
  
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
                  "verbose"    : False
                  }`
  
  (See run_BISIP.py for an example script on how to use the inversion function and plot results)

**3. References**

This code uses part of the PyMC package (https://github.com/pymc-devs/pymc)

<sub><sup>
Chen, Jinsong, Andreas Kemna, and Susan S. Hubbard. 2008. “A Comparison between
    Gauss-Newton and Markov-Chain Monte Carlo–based Methods for Inverting
    Spectral Induced-Polarization Data for Cole-Cole Parameters.” Geophysics
    73 (6): F247–59. doi:10.1190/1.2976115.
    
Dias, Carlos A. 2000. “Developments in a Model to Describe Low-Frequency
    Electrical Polarization of Rocks.” Geophysics 65 (2): 437–51.
    doi:10.1190/1.1444738.
    
Gamerman, Dani, and Hedibert F. Lopes. 2006. Markov Chain Monte Carlo:
    Stochastic Simulation for Bayesian Inference, Second Edition. CRC Press.
    
Ghorbani, A., C. Camerlynck, N. Florsch, P. Cosenza, and A. Revil. 2007.
    “Bayesian Inference of the Cole–Cole Parameters from Time- and Frequency-
    Domain Induced Polarization.” Geophysical Prospecting 55 (4): 589–605.
    doi:10.1111/j.1365-2478.2007.00627.x.
    
Hoff, Peter D. 2009. A First Course in Bayesian Statistical Methods. Springer
    Science & Business Media.
    
Keery, John, Andrew Binley, Ahmed Elshenawy, and Jeremy Clifford. 2012.
    “Markov-Chain Monte Carlo Estimation of Distributed Debye Relaxations in
    Spectral Induced Polarization.” Geophysics 77 (2): E159–70.
    doi:10.1190/geo2011-0244.1.
    
Nordsiek, Sven, and Andreas Weller. 2008. “A New Approach to Fitting Induced-
    Polarization Spectra.” Geophysics 73 (6): F235–45. doi:10.1190/1.2987412.
    
Patil, A., D. Huard and C.J. Fonnesbeck. 2010. PyMC: Bayesian Stochastic
    Modelling in Python. Journal of Statistical Software, 35(4), pp. 1-81
    
Pelton, W. H., W. R. Sill, and B. D. Smith. 1983. Interpretation of Complex
    Resistivity and Dielectric Data — Part 1. Vol 29. Geophysical Transactions.
    
Pelton, W. H., S. H. Ward, P. G. Hallof, W. R. Sill, and P. H. Nelson. 1978.
    “Mineral Discrimination and Removal of Inductive Coupling with
    Multifrequency IP.” Geophysics 43 (3): 588–609. doi:10.1190/1.1440839.
    
Shin, S. W., S. Park, and D. B. Shin. 2015. “Development of a New Equivalent
    Circuit Model for Spectral Induced Polarization Data Analysis of Ore
    Samples.” Environmental Earth Sciences 74 (7): 5711–16.
    doi:10.1007/s12665-015-4588-z.
<sub><sup>
