# BISIP
## Bayesian inference of spectral induced polarization parameters (Python 2.7)
### Download the latest executables for [Mac OS X (64bit)](https://www.dropbox.com/s/6namw3yiq9we9k2/BISIP%20Workplace%20%28OS%20X%2064bit%29.zip?dl=1) or [Windows (64bit)](https://github.com/clberube/BISIP/releases/download/v1.0-windows/BISIP_Workplace_windows.zip)

Charles L. Bérubé, Michel Chouteau, Pejman Shamsipour, Randolph J. Enkin, Gema R. Olivo, Bayesian inference of spectral induced polarization parameters for laboratory complex resistivity measurements of rocks and soils, In Computers & Geosciences, Volume 105, 2017, Pages 51-64, ISSN 0098-3004, https://doi.org/10.1016/j.cageo.2017.05.001.
(http://www.sciencedirect.com/science/article/pii/S0098300416306446)

### Contents
**1. Using the standalone executables**  
**2. Cloning the repository and installing dependencies**  
**3. Starting the GUI from the command prompt**  
**4. Calling the inversion function from another script**  
**5. Building the standalone GUI executables**  
**6. Building the BISIP_cython_funcs.pyd (Windows) or BISIP_cython_funcs.so (OS X) files**  
**7. References**  

================================================================================================================================

**1. Using the standalone executables**
  
  No Python installation is needed to run, the applications are standalone  
  Executables were compiled on OS X 10.11.6 and Windows 10
  
    a. Extract the BISIP Workplace folder to a local directory  
    b. Launch the executable in BISIP Workplace  
    c. A terminal window will open (Allow a few seconds/minutes to load Python if using Windows)
    d. Import example data files and launch inversions using the default MCMC parameters  
    e. Results are saved in subfolders inside the BISIP Workplace folder

**2. Cloning the repository and installing dependencies**

  Clone or Download repository to local folder  
  Dependencies: Python 2.7, NumPy, matplotlib, SciPy, PyMC, Tkinter  
  PyMC (<https://github.com/pymc-devs/pymc>) is installed with any of the following:
  
    conda install pymc
    pip install git+git://github.com/pymc-devs/pymc
    easy_install pymc
  
  Other dependencies are automatically installed with most Python distributions
      
**3. Starting the GUI from the command prompt**
   
  Open a terminal to the local BISIP directory and enter:
 
    python BISIP_GUI.py
  
**4. Calling the inversion function from another script**

  Import only the inversion function using:
  
    from BISIP_models import mcmcSIPinv
  
  And obtain results using all default arguments and MCMC parameters with:
  
    sol = mcmcSIPinv('ColeCole', '/Documents/DataFiles/DATA.dat')
  
  To call the function with optional arguments:
  
  Example for Debye decomposition: 
    
    sol = mcmcSIPinv( model='PDecomp', filename='/Documents/DataFiles/DATA.dat', 
                      headers=1, ph_units='mrad', mcmc=mcmc_params, adaptive=True,  
                      debye_poly=4, c_exp = 1.0, keep_traces=False)
                      
  Example for Cole-Cole inversion:
    
    sol = mcmcSIPinv( model='ColeCole', filename='/Documents/DataFiles/DATA.dat', 
                      headers=1, ph_units='mrad', mcmc=mcmc_params, adaptive=False,  
                      cc_modes=2, keep_traces=False)
  
  Where `mcmc_params` is a python dictionary.
    
    mcmc_p = {"adaptive"   : True,
              "nb_chain"   : 1,
              "nb_iter"    : 500000,
              "nb_burn"    : 400000,
              "thin"       : 1,
              "tune_inter" : 10000,    # Only used when mcmc_p["adaptive"]=False
              "prop_scale" : 1.0,      # Only used when mcmc_p["adaptive"]=False
              "verbose"    : False,
              "cov_inter"  : 50000,    # Only used when mcmc_p["adaptive"]=True
              "cov_delay"  : 50000,    # Only used when mcmc_p["adaptive"]=True
              }
                  
  And `sol` is a self-explanatory python dictionary containing the results:
  
    In []: sol.keys()
    Out[]: ['pymc_model',
           'params',
           'fit',
           'model_type',
           'path',
           'data',
           'mcmc',
           'SIP_model']
  
  For example, to return the optimal parameters of a Double Cole-Cole model (R0, c1, c2, m1, m2, tau1, tau2):
  
    In []: sol['params']
    Out[]: {'R0': 51467.05483286261,
           'R0_std': 126.18837609979391,
           'c': array([2.127E-01, 5.805E-01]),
           'c_std': array([5.864E-03, 5.611E-03]),
           'm': array([1.435E-01, 9.887E-01]),
           'm_std': array([2.895E-03, 9.175E-03]),
           'tau': array([1.267E+01, 1.692E-06]),
           'tau_std': array([7.253E-01, 2.692E-08])}
  
    In []: sol['params']['c']
    Out[]: array([2.127E-01, 5.805E-01])
  
  To return the MCMC parameters that were used in the inversion:
  
    In []: sol["mcmc"]
    Out[]: {'adaptive': True,
           'cov_delay': 5000,
           'cov_inter': 5000,
           'nb_burn': 0,
           'nb_chain': 1,
           'nb_iter': 30000,
           'prop_scale': 1.0,
           'thin': 1,
           'tune_inter': 10000,
           'verbose': False}
  
  To return the most probable fit:
  
    In []: sol['fit']['best']
    Out[]: array([ 74597.52558689-51642.13051532j,   93161.78463202-47966.09987357j,
                  114306.00769672-40047.86385792j,  130011.45140055-31266.24589905j,
                  141037.77202291-23556.40715467j,  148726.78729011-17693.58212181j,
                  154298.97343568-13715.72484526j,  158636.12650627-11383.13733406j,
                  162542.62504439-10328.76045588j,  166394.88440487-10315.84443450j,
                  170630.34483685-11057.19749671j,  175545.93354632-12264.58908308j,
                  181309.50373747-13593.27424061j,  187893.01650757-14645.93643453j,
                  195018.29894754-15052.77811872j,  202194.30663152-14608.94172405j,
                  208878.89960888-13367.96063622j,  214673.56872037-11595.75771159j,
                  219414.69115156 -9622.59282482j,  223136.12001005 -7714.70066947j,
                  225977.95223702 -6028.20247440j,  228111.44641247 -4623.37088204j])

To return the raw data:

    In []: sol['data']['Z']
    Out[]: array([ 70409.92224316-59566.43355993j,   92155.36514813-51271.53207947j,
                  112377.54681250-38393.49877798j,  125216.20900653-28188.83940559j,
                  134110.39775196-21264.63487700j,  140950.11273197-16854.94020560j,
                  146799.44963326-13978.74885751j,  152007.69171323-12459.91437639j,
                  156629.40293082-11484.14222119j,  161202.23795165-11420.09275102j,
                  171100.34202259-10458.80049734j,  175794.43251825-10979.19172297j,
                  178848.45995427-12658.91499277j,  183653.19719158-14360.61529841j,
                  190138.06667213-15416.72216746j,  197532.07600391-15226.31555521j,
                  204703.73156468-13802.76032524j,  211153.63099728-11636.01274762j,
                  217279.66579033 -9154.96275503j,  225820.90718799 -6727.65204251j,
                  232480.53201552 -5313.6790781j ,  242925.22993713 -4502.38295756j])
  
  Plotting the comparison
  
    import matplotlib.pyplot as plt
    plt.plot(sol['data']['Z'].real, sol['data']['Z'].imag)
    plt.plot(sol['fit']['best'].real, sol['fit']['best'].imag)

  The inversion function also returns the full PyMC object:
  
    In []: sol['pymc_model']
    Out[]: <pymc.MCMC.MCMC at 0x11e37a110>
  
  See run_BISIP.py for an example script on how to use the inversion function and plot results  
  See the PyMC documentation (<https://pymc-devs.github.io/pymc/>) to extract information from the PyMC object

**Data must be formatted using the following template:**  

    Frequency, Amplitude  , Phase shift , Amplit error, Phase error  
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

**5. Building the standalone GUI executables**

Install pyinstaller with:  
```sh
pip install pyinstaller
```  

Open a terminal to the directory where BISIP_GUI.py and BISIP_GUI.spec are located  
Enter the following:  
```sh
pyinstaller BISIP_GUI_win.spec
```  
Or  
```sh
pyinstaller BISIP_GUI_osx.spec
```
On Windows this works best with pyinstaller 3.1, setuptools 19.2, pymc 2.3.5 and numpy 1.11.3.

**6. Building the BISIP_cython_funcs.pyd (Windows) or BISIP_cython_funcs.so (OS X) files**

Install cython with:  
```sh
conda install cython  
```

Open a terminal to the directory where `BISIP_cython_funcs.pyx` and `cython_setup.py` are located  
Enter the following:  
```sh
python cython_setup.py build_ext --inplace
```

**7. References**

<sub>Charles L. Bérubé, Michel Chouteau, Pejman Shamsipour, Randolph J. Enkin, and Gema R. Olivo. 2017. Bayesian inference of spectral induced polarization parameters for laboratory complex resistivity measurements of rocks and soils. Computers & Geosciences 105: 51-64, ISSN 0098-3004, https://doi.org/10.1016/j.cageo.2017.05.001.

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

<sub>Pelton, W. H., S. H. Ward, P. G. Hallof, W. R. Sill, and P. H. Nelson. 1978.
    “Mineral Discrimination and Removal of Inductive Coupling with
    Multifrequency IP.” Geophysics 43 (3): 588–609. doi:10.1190/1.1440839.

<sub>Pelton, W. H., W. R. Sill, and B. D. Smith. 1983. Interpretation of Complex
    Resistivity and Dielectric Data — Part 1. Vol 29. Geophysical Transactions.
