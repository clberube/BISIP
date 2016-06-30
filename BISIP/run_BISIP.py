# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 12:05:22 2015

@author:    clafreniereberube@gmail.com
            École Polytechnique Montréal

Copyright (c) 2015-2016 Charles L. Bérubé

"""

import numpy as np
from BISIP_models import mcmcSIPinv
import BISIP_invResults as iR

#==============================================================================
""" 1.
    Model to use ?"""
# ex: model = "ColeCole", "Dias", "Debye", "Shin"
model = "ColeCole"
#model = "Dias"
#model = "Debye"
#model = "BDebye"
#model = "Shin"

#==============================================================================
""" 2.
    Markov-chain Monte-Carlo parameters ?"""
mcmc_params = {'nb_chain'   : 1,
               'nb_iter'    : 10000,
               'nb_burn'    : 9000,
               'thin'       : 1,
               'prop_scale' : 1,
               'tune_inter' : 1000,
               'verbose'    : False,
               }

#==============================================================================
""" 3.
    Paths to files ?"""
filename = [
#            "/Users/Charles/Documents/SIP dat files/SIP-MLA12_K389005_stable.dat",
#            "/Users/Charles/Documents/SIP dat files/SIP-MLA12_K389019_stable.dat",
#            "/Users/Charles/Documents/SIP dat files/SIP-MLA12_K389046_stable.dat",
#            "/Users/Charles/Documents/SIP dat files/SIP-MLA12_K389055_stable.dat",
#            "/Users/Charles/Documents/SIP dat files/SIP-MLA12_K389058_stable.dat",
#            "/Users/Charles/Documents/SIP dat files/SIP-MLA12_K389062_stable.dat",
#            "/Users/Charles/Documents/SIP dat files/SIP-MLA12_K389077_stable.dat",
#            "/Users/Charles/Documents/SIP dat files/SIP-MLA12_K389198_stable.dat",
#            "/Users/Charles/Documents/SIP dat files/SIP-MLA12_K389214_stable.dat",
#            "/Users/Charles/Documents/SIP dat files/SIP-MLA12_K389216_stable.dat",
#            "/Users/Charles/Documents/SIP dat files/SIP-MLA12_K389219_stable.dat",
#            "/Users/Charles/Documents/SIP dat files/SIP-MLA12_K389227_stable.dat",
            "/Users/Charles/Documents/SIP dat files/SIP-Reciprocals_K389175_avg.dat",
            ]

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
    print '\nReading file:', fn, '(#%d/%d)' %(i+1,len(filename))
    sol = mcmcSIPinv(model, mcmc_params, fn, headers=skip_header, ph_units=ph_units, debye_poly=4, keep_traces=False)

    """Plot fit and data ?"""
    if True:
        fig_fit = iR.plot_fit(sol["data"], sol["fit"], model, fn, save=True)

    """Save results ?"""
    if False:
        iR.save_resul(sol["pymc_model"], sol["params"], model, fn)

    """Plot Debye relaxation time distribution ?"""
    if False:
        fig_RTD = iR.plot_debye(sol, fn, save=False, draw=True)

    """Print numerical results ?"""
    if True:
        iR.print_resul(sol["params"], model, fn)

    """Plot parameter histograms ?"""
    if False:
        fig_histo = iR.plot_histo(sol["pymc_model"], model, fn, False)

    if False:
        fig_trace = iR.plot_traces(sol["pymc_model"], model, fn, False)

    """Plot parameter summary and Gelman-Rubin convergence test ?"""
    if False:
        fig_summary = iR.plot_summary(sol["pymc_model"])

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
Pelton, W. H., W. R. Sill, and B. D. Smith. 1983. Interpretation of Complex
    Resistivity and Dielectric Data — Part 1. Vol 29. Geophysical Transactions.
Pelton, W. H., S. H. Ward, P. G. Hallof, W. R. Sill, and P. H. Nelson. 1978.
    “Mineral Discrimination and Removal of Inductive Coupling with
    Multifrequency IP.” Geophysics 43 (3): 588–609. doi:10.1190/1.1440839.
"""