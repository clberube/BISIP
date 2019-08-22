# #!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: charles
# @Date:   22-08-2019
# @Email:  charles@goldspot.ca
# @Last modified by:   charles
# @Last modified time: 22-08-2019


import sys
sys.path.append('/Users/charles/Repositories/BISIP')

import matplotlib.pyplot as plt
from bisip import mcmcinv


model = 'PDecomp'
src_path = '/Users/charles/Repositories/BISIP/data files/SIP-K389170_avg.dat'
mcmc = {'adaptive': True,
        'nb_chain': 1,
        'nb_iter': 100000,
        'nb_burn': 80000,
        'thin': 1,
        'tune_inter': 500,
        'prop_scale': 1.0,
        'verbose': False,
        'cov_inter': 1000,
        'cov_delay': 1000,
        }

sol = mcmcinv(model, src_path, mcmc=mcmc)

sol.plot_rtd()
sol.plot_fit(save=True, save_as_png=True, dpi=144)
