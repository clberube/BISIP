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
model = "lam"

#==============================================================================
""" 2.
    Markov-chain Monte-Carlo parameters ?"""
mcmc_p = {"adaptive"   : False,
          "nb_chain"   : 1,
          "nb_iter"    : 100,
          "nb_burn"    : 0,
          "thin"       : 1,
          "tune_inter" : 10,
          "prop_scale" : 1.0,
          "verbose"    : False, 
          "cov_inter"  : 100,
          "cov_delay"  : 100,
          }

#==============================================================================
""" 3.
    Paths to files ?"""
    
reflist = os.listdir("/Users/Charles/Documents/SIP dat files")
reflist = [x for x in reflist if not x.startswith('.')]
reflist = [x for x in reflist if ("AVG" in x)]

reflist = [x for x in reflist if "MLA12" in x]
reflist = [x for x in reflist if "_stable" in x]
reflist = [reflist[x] for x in [2,6,9,11]] # RockTypes

reflist = [reflist[1]]

filename = ["/Users/Charles/Documents/SIP dat files/"+x for x in reflist]

fn = filename[0]

#==============================================================================
""" 4.
    Number of headers to skip ?"""
skip_header = 1

#==============================================================================
""" 5.
    Phase units in raw data file ?"""
# {"rad" = radians}  {"mrad" = milliradians}  {"deg" = degrés}
ph_units = "mrad"

sol = mcmcinv(model, fn, mcmc=mcmc_p, headers=skip_header, 
                       ph_units=ph_units, decomp_poly=4, cc_modes=2, 
                       c_exp=1.0, log_min_tau=-3, guess_noise=False, 
                       keep_traces=False, ccdt_cfg=None)
