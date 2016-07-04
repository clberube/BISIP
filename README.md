# BISIP
## Bayesian inference of spectral induced polarization parameters (Python 2.7)

**1. Installation**

  Clone or Download repository to local folder

  Install Python dependencies (Python 2.7, NumPy, matplotlib, SciPy, PyMC, Tkinter)
  
  **_OR_**
  
  Download the executables (OS X and Windows) directly at:  
  <https://drive.google.com/open?id=0B3_1MlzD_zfQdUw5NTVMOUZVTXM>
  
**2. Usage**
  
  Run BISIP_GUI.py to start the GUI
  
  **_OR_**

  Import the inversion function using:
  
    from BISIP_models import mcmcSIPinv
  
  And obtain results using all default arguments and MCMC parameters with:
  
    sol = mcmcSIPinv("ColeCole", "/Documents/DataFiles/DATA.dat")
  
  The full list of optional arguments is:
  
    sol = mcmcSIPinv( model="ColeCole", filename="/Documents/DataFiles/DATA.dat", 
                      mcmc=mcmc_params, headers=1, ph_units="mrad", cc_modes=2, 
                      debye_poly=4, keep_traces=False)
  
  Where `mcmc_params` is a python dictionary:
  
    mcmc_params = {"nb_chain"  : 1,
                  "nb_iter"    : 50000,
                  "nb_burn"    : 40000,
                  "thin"       : 1,
                  "tune_inter" : 1000,
                  "prop_scale" : 1.0,
                  "verbose"    : False
                  }
  
  And`sol` is a self-explanatory python dictionary containing the results:
  
    In []: sol.keys()
    Out[]: ['pymc_model', 'params', 'fit', 'data']
  
  (See run_BISIP.py for an example script on how to use the inversion function and plot results)

**Data must be formatted using the following template:**  

    Freq (Hz), Res (Ohm-m),  Phase (deg), dRes (Ohm-m), dPhase (deg)  
    6.000e+03, 1.17152e+05, -2.36226e+02, 1.171527e+01, 9.948376e-02  
    3.000e+03, 1.22177e+05, -1.46221e+02, 1.392825e+01, 1.134464e-01  
    1.500e+03, 1.25553e+05, -9.51099e+01, 2.762214e+01, 2.199114e-01  
    ........., ..........., ............, ............, ............  
    ........., ..........., ............, ............, ............  
    ........., ..........., ............, ............, ............  
    4.575e-02, 1.66153e+05, -1.21143e+01, 1.947314e+02, 1.171115e+00  
    2.288e-02, 1.67988e+05, -9.36718e+00, 3.306003e+02, 1.9669860+00  
    1.144e-02, 1.70107e+05, -7.25533e+00, 5.630541e+02, 3.310889e+00

Save data in .csv, .txt, .dat, ... extension file  
Comma separation between columns is mandatory  
Column order is very important  
Frequency, Amplitude, Phase, Error of Amplitude, Error of Phase  
Phase units may be milliradians, radians or degrees  
Units are specified in main GUI window or as function argument (e.g. `ph_units="mrad"`)  
Amplitude units may be Ohm-m or Ohm  
A number of header lines may be skipped in the main GUI window or as function argument (e.g. `headers=1`)  
In this example Nb header lines = 1  
To skip high-frequencies, increase Nb header lines  
Scientific or standard notation is OK  

**3. Building the standalone GUI executable**

Install pyinstaller `pip install pyinstaller`  
Open a terminal to the directory where BISIP_GUI.py is located  
Enter the following:

```sh
pyinstaller --hidden-import=scipy.linalg.cython_blas --hidden-import=scipy.linalg.cython_lapack --hidden-import=scipy.special._ufuncs_cxx --onefile BISIP_GUI.py
```

**4. Building the BISIP_cython_funcs.pyd or BISIP_cython_funcs.so file**

Install cython `conda install cython`  
Open a terminal to the directory where `BISIP_cython_funcs.pyx` and `cython_setup.py` are located  
Enter the following:

```sh
python cython_setup.py build_ext --inplace
```
**5. References**

This code uses part of the PyMC package (https://github.com/pymc-devs/pymc)

<sub>Chen, Jinsong, Andreas Kemna, and Susan S. Hubbard. 2008. “A Comparison between
    Gauss-Newton and Markov-Chain Monte Carlo–based Methods for Inverting
    Spectral Induced-Polarization Data for Cole-Cole Parameters.” Geophysics
    73 (6): F247–59. doi:10.1190/1.2976115.
    
<sub>Dias, Carlos A. 2000. “Developments in a Model to Describe Low-Frequency
    Electrical Polarization of Rocks.” Geophysics 65 (2): 437–51.
    doi:10.1190/1.1444738.
    
<sub>Gamerman, Dani, and Hedibert F. Lopes. 2006. Markov Chain Monte Carlo:
    Stochastic Simulation for Bayesian Inference, Second Edition. CRC Press.
    
<sub>Ghorbani, A., C. Camerlynck, N. Florsch, P. Cosenza, and A. Revil. 2007.
    “Bayesian Inference of the Cole–Cole Parameters from Time- and Frequency-
    Domain Induced Polarization.” Geophysical Prospecting 55 (4): 589–605.
    doi:10.1111/j.1365-2478.2007.00627.x.
    
<sub>Hoff, Peter D. 2009. A First Course in Bayesian Statistical Methods. Springer
    Science & Business Media.
    
<sub>Keery, John, Andrew Binley, Ahmed Elshenawy, and Jeremy Clifford. 2012.
    “Markov-Chain Monte Carlo Estimation of Distributed Debye Relaxations in
    Spectral Induced Polarization.” Geophysics 77 (2): E159–70.
    doi:10.1190/geo2011-0244.1.
    
<sub>Nordsiek, Sven, and Andreas Weller. 2008. “A New Approach to Fitting Induced-
    Polarization Spectra.” Geophysics 73 (6): F235–45. doi:10.1190/1.2987412.
    
<sub>Patil, A., D. Huard and C.J. Fonnesbeck. 2010. PyMC: Bayesian Stochastic
    Modelling in Python. Journal of Statistical Software, 35(4), pp. 1-81
    
<sub>Pelton, W. H., W. R. Sill, and B. D. Smith. 1983. Interpretation of Complex
    Resistivity and Dielectric Data — Part 1. Vol 29. Geophysical Transactions.
    
<sub>Pelton, W. H., S. H. Ward, P. G. Hallof, W. R. Sill, and P. H. Nelson. 1978.
    “Mineral Discrimination and Removal of Inductive Coupling with
    Multifrequency IP.” Geophysics 43 (3): 588–609. doi:10.1190/1.1440839.
    
