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

#    from models import mcmcinv
from bisip.models import mcmcinv
import lib_dd.config.cfg_single as cfg_single

#import bisip.invResults as iR
import pickle as pickle
import os

import matplotlib as mpl
mpl.rc_file_defaults()

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
#model = "PDecomp"
#model = "DDebye"
#model = "Shin"
model = "CCD"

#==============================================================================
""" 2.
    Markov-chain Monte-Carlo parameters ?"""
mcmc_p = {"adaptive"   : True,
          "nb_chain"   : 1,
          "nb_iter"    : 10000,
          "nb_burn"    : 8000,
          "thin"       : 1,
          "tune_inter" : 10000,
          "prop_scale" : 1.0,
          "verbose"    : False, 
          "cov_inter"  : 1000,
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
    reflist = [x for x in reflist if ("AVG" in x)]
#    reflist = [x for x in reflist if ("_avg" in x)]
#    reflist = [x for x in reflist if ("Reciprocals" in x) or ("MLA12" in x)]
#    reflist = [x for x in reflist if "Reciprocals" in x]
    reflist = [x for x in reflist if "MLA12" in x]
    reflist = [x for x in reflist if "_stable" in x]
#    reflist = [x for x in reflist if "55" in x]
#    reflist = [reflist[75]]
#    reflist = [reflist[1]]
#    reflist = reflist[1:5]
#    reflist = ["SIP-K389055.dat"]
#    reflist = [reflist[x] for x in [2,0,3]]
#    reflist = [reflist[x] for x in [4,1,-2]]
#    reflist = [reflist[x] for x in [4,0,1,5,3,-2]]
    
#    reflist = [reflist[x] for x in [2,4,5,9,10]]

#    reflist = [reflist[x] for x in [2,6,7,9,11]] # RockTypes
#    reflist = [reflist[x] for x in [2,6,9,11]] # RockTypes

    reflist = [reflist[x] for x in [2,4,5,0,3,1,10]]

#    reflist = [reflist[x] for x in [2, 6, -3,-1]]
#    reflist = [reflist[x] for x in [77]]
#    reflist = [reflist[1]]

#    reflist = ["703","709","710","737","738","715","717","723","724","730","732","735"]
#    reflist = ['AVG_SIP-Reciprocals_K389%s.dat'%x for x in reflist]

    filename = ["/Users/Charles/Documents/SIP dat files/"+x for x in reflist]
#    filename = ["/Users/Charles/Documents/SIP dat files/AVG_SIP-Reciprocals_K389369.dat"]
#    filename = [
#            '/Users/Charles/Documents/SIP dat files/SIP-Bravo_11mhz_test_2_sandstones_nord_avg.dat',
##            '/Users/Charles/Documents/SIP dat files/SIP-Bravo_11mhz_test_5_sandstones_sud_avg.dat',
#            '/Users/Charles/Documents/SIP dat files/SIP-Bravo_11mhz_test_3_mudstones_avg.dat',
#            '/Users/Charles/Documents/SIP dat files/SIP-Bravo_11mhz_test_4_sandstonesGOLD_avg.dat',
#            ]
#    filename = ['/Users/Charles/Documents/SIP dat files/SIP-BravoProfile_Station%s.dat' %str(x).zfill(2) for x in range(29)]
#    reflist = [f.split("/")[-1] for f in filename]

#    filename = [
#                "/Users/Charles/Documents/SIP dat files/B7-GB-semaine1-degree-ohm.csv",
#                "/Users/Charles/Documents/SIP dat files/B7-GB-semaine12-degree-ohm.csv",
#                "/Users/Charles/Documents/SIP dat files/B1-semaine17-degree-ohm.csv",
#                ]

    #==============================================================================
    """ 4.
        Number of headers to skip ?"""
    skip_header = 1
    
    #==============================================================================
    """ 5.
        Phase units in raw data file ?"""
    # {"rad" = radians}  {"mrad" = milliradians}  {"deg" = degrés}
    ph_units = "mrad"
    
    config = cfg_single.cfg_single()
    config['nr_terms_decade'] = 20
    config['lambda'] = 20
    config['norm'] = 10
    
    #==============================================================================
    # Call to the inversion function for every file
    for i, fn in enumerate(filename):
#    for i in range(repeat):
#        fn = filename[0]
        print('\nReading file:', fn, '(#%d/%d)' %(i+1,len(filename)))
        sol.append(mcmcinv(model, fn, mcmc=mcmc_p, headers=skip_header, 
                           ph_units=ph_units, decomp_poly=4, cc_modes=2, 
                           c_exp=1.0, log_min_tau=-3, guess_noise=False, 
                           keep_traces=False, ccdt_cfg=None))
    
        """Plot fit and data ?"""
        sol[i].plot_fit(save=True, draw=False)
                
        """Save results ?"""
        sol[i].save_results()
    
        """Plot Debye relaxation time distribution ?"""
#        rtd = sol[i].plot_rtd(save=True, draw=False)
    
        """Print numerical results ?"""
#        sol[i].print_results()
    
        """Plot parameter histograms ?"""
#        sol[i].plot_histograms(save=True)
#        sol[i].plot_traces(save=True)
        
#        """Plot parameter summary and Gelman-Rubin convergence test ?"""
#        if False:
#            fig_kde = iR.plot_KDE(sol, "a0", "a1", save=False)

#sol = sol[0]
#sol.merge_results([x.split(".")[0] for x in reflist])

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