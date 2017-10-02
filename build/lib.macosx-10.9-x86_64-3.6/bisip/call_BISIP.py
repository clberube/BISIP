# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 12:05:22 2015
@author:    charleslberube@gmail.com
            École Polytechnique Montréal
Copyright (c) 2015-2016 Charles L. Bérubé
"""
from __future__ import print_function

from future import standard_library
standard_library.install_aliases()
from builtins import str
from builtins import range

from models import mcmcinv
import invResults as iR
import pickle as pickle
import os

def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, -1)

#==============================================================================
""" 1.
    Model to use ?"""
# ex: model = "ColeCole", "Dias", "Debye", "Shin"
#model = "ColeCole"
#model = "Dias"
#model = "PDebye"
model = "PDecomp"
#model = "DDebye"
#model = "Shin"

#==============================================================================
""" 2.
    Markov-chain Monte-Carlo parameters ?"""
mcmc_p = {"adaptive"   : True,
          "nb_chain"   : 1,
          "nb_iter"    : 15000,
          "nb_burn"    : 13000,
          "thin"       : 1,
          "tune_inter" : 10000,
          "prop_scale" : 1.0,
          "verbose"    : False,
          "cov_inter"  : 500,
          "cov_delay"  : 1000,
          }
sol = []

#for noise in [10, 5, 1]:
#for noise in range(1,11):
for noise in [1]:
#noise = 5
    adapt = True
#    adapt = False
    repeat = 1
    save_as = "%dmrad_%s_%d_MCMC_Solutions_Adaptive_%s.pkl" %(noise,model,repeat,str(adapt))
    #save_as = "%dmrad_%d_MCMC_Traces_Adaptive_%s.pkl" %(noise,repeat,str(adapt))
    #save_as = "tests.pkl"
    
    #==============================================================================
    """ 3.
        Paths to files ?"""
        
    reflist = os.listdir("/Users/Charles/Documents/SIP dat files")
    reflist = [x for x in reflist if not x.startswith('.')]
    reflist = [x for x in reflist if "AVG" in x]
    reflist = [x for x in reflist if "Reciprocals" in x]
    
    reflist = [reflist[0]]
    
    filename = ["/Users/Charles/Documents/SIP dat files/"+x for x in reflist]
    
    #==============================================================================
    """ 4.
        Number of headers to skip ?"""
    skip_header = 3
    
    #==============================================================================
    """ 5.
        Phase units in raw data file ?"""
    # {"rad" = radians}  {"mrad" = milliradians}  {"deg" = degrés}
    ph_units = "mrad"
    
    #==============================================================================
    # Call to the inversion function for every file
    for i, fn in enumerate(filename):
#    for i in range(repeat):
#        fn = filename[0]
        print('\nReading file:', fn, '(#%d/%d)' %(i+1,len(filename)))
        from models import mcmcinv
        sol.append(mcmcinv(model, fn, mcmc=mcmc_p, headers=skip_header, ph_units=ph_units, decomp_poly=4, cc_modes=2, c_exp=1.0, log_min_tau=-3, keep_traces=False))
    
        """Plot fit and data ?"""
        if True:
            fig_fit = iR.plot_fit(sol[i], save=True)
    
        """Save results ?"""
        if True:
            iR.save_resul(sol[i])
    
        """Plot Debye relaxation time distribution ?"""
        if True:
            fig_RTD = iR.plot_rtd(sol[i], save=True, draw=False)
    
        """Print numerical results ?"""
        if False:
            iR.print_resul(sol[i])
    
        """Plot parameter histograms ?"""
        if True:
            fig_histo = iR.plot_histo(sol[i], save=True)
    
        if True:
            fig_trace = iR.plot_traces(sol[i], save=True)
    
        """Plot parameter summary and Gelman-Rubin convergence test ?"""
        if False:
            fig_kde = iR.plot_KDE(sol, "a0", "a1", save=False)


iR.merge_results(sol[0], [x.split(".")[0] for x in reflist])

# For further use in Python
#saved_sol = [{key: value for key, value in list(s.items()) if key not in ["pymc_model"]} for s in sol]
#save_object(saved_sol, save_as)
#print "Solutions saved in list under", save_as

#==============================================================================
#===========================DATA FILE TEMPLATE=================================
#==============================================================================

"""
FORMAT DES DONNÉES
Mettre les données dans un fichier .csv, .txt, ou .dat
La séparation des colonnes par une virgule est importante
Les unités de la phase peuvent être des miliradians, radians ou degrés
Les unités de l'amplitude peuvent être des Ohm ou Ohm-m
(À spécifier dans la section plus haut, ici dans l'exemple skip_header = 1)
La colonne 'Courant (A)' n'est pas obligatoire
Les données doivent être sous le format suivant :
=============================================================================
Freq (Hz), Res (Ohm-m),  Phase (deg), dRes (Ohm-m), dPhase (deg), Current (A)
9.600e+03, 1.02676e+05, -1.29148e+02, 1.398449e+02, 1.361356e+00, 1.57806e-05
6.000e+03, 1.05713e+05, -9.52791e+01, 2.790833e+01, 2.635447e-01, 1.54739e-05
3.752e+03, 1.07753e+05, -7.10489e+01, 3.857578e+01, 3.577924e-01, 8.15439e-06
2.345e+03, 1.09314e+05, -5.39818e+01, 2.168789e+02, 1.984439e+00, 2.66668e-06
........., ..........., ............, ............, ............, ...........
........., ..........., ............, ............, ............, ...........
........., ..........., ............, ............, ............, ...........
7.584e-02, 1.23939e+05, -1.37357e+01, 3.042719e+02, 2.455678e+00, 8.20719e-06
4.740e-02, 1.24433e+05, -1.35053e+01, 4.442275e+01, 3.577924e-01, 7.94366e-06
2.962e-02, 1.24935e+05, -1.30672e+01, 4.100396e+02, 3.281218e+00, 6.95748e-06
=============================================================================
"""

#==============================================================================
#================================REFERENCES====================================
#==============================================================================
"""
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
    doi:10.1007/s12665-015-4588-z."""