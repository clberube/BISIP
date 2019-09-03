# #!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: charles
# @Date:   22-08-2019
# @Email:  charles@goldspot.ca
# @Last modified by:   cberube
# @Last modified time: 03-09-2019


import matplotlib.pyplot as plt
from bisip import mcmcinv


model = 'PDecomp'
src_path = '/Users/cberube/Repositories/BISIP/data files/SIP-K389170_avg.dat'
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

sol = mcmcinv(model, src_path, mcmc=mcmc, guess_noise=False)
sol.fit()

sol.plot_rtd()
sol.plot_fit()
sol.plot_histograms()
sol.plot_KDE('total_m', 'log_mean_tau')
sol.plot_traces()
sol.plot_summary()
sol.save_results()
