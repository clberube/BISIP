#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: clberube
# @Date:   12-03-2020
# @Email:  charles.lafreniere-berube@polymtl.ca
# @Last modified by:   charles
# @Last modified time: 2020-03-18T16:01:58-04:00


import os
import warnings

import numpy as np
import matplotlib.pyplot as plt

# import bisip
# from bisip import PolynomialDecomposition, PeltonColeCole, Dias2000
#
#
# def run_test(dias=True, colecole=True, debye=True):
#
#     fp = f'data/SIP-K389175.dat'
#     fp = os.path.join(os.path.dirname(bisip.__file__), fp)
#
#     if colecole:
#         print('Testing ColeCole model')
#         model = PeltonColeCole(fp, nwalkers=32, n_modes=2, nsteps=1000)
#         model.fit()
#
#         # Get the mean parameter values and their std
#         # discarding the first 1000 steps (burn-in)
#         values = model.get_param_mean(discard=800)
#         uncertainties = model.get_param_std(discard=800)
#
#         for n, v, u in zip(model.param_names, values, uncertainties):
#             print(f'{n}: {v:.5f} +/- {u:.5f}')
#
#     if dias:
#         print('Testing Dias model')
#         model = Dias2000(fp, nwalkers=32, nsteps=2000)
#         start = np.vstack([[1.0, 0.25, -10, 5, 0.5] for _ in range(32)])
#         start += 1e-1*start*(np.random.rand(*start.shape) - 1)
#         # Update parameter boundaries inplace
#         model.fit(p0=start)
#         chain = model.get_chain(discard=1000, thin=1, flat=True)
#         # Get the mean parameter values and their std
#         # discarding the first 1000 steps (burn-in)
#         values = model.get_param_mean(chain)
#         uncertainties = model.get_param_std(chain)
#
#         for n, v, u in zip(model.param_names, values, uncertainties):
#             print(f'{n}: {v:.5f} +/- {u:.5f}')
#
#     if debye:
#         print('Testing Debye Decomposition')
#         model = PolynomialDecomposition(fp, nwalkers=32, poly_deg=4,
#                                         nsteps=2000)
#         # Update parameter boundaries inplace
#         model.params.update(a0=[-2, 2])
#         model.fit()
#
#         chain = model.get_chain(discard=1000, thin=1, flat=True)
#
#         # Get the mean parameter values and their std
#         # discarding the first 1000 steps (burn-in)
#         values = model.get_param_mean(chain)
#         uncertainties = model.get_param_std(chain)
#
#         for n, v, u in zip(model.param_names, values, uncertainties):
#             print(f'{n}: {v:.5f} +/- {u:.5f}')
#
#     print('Testing plotlib with last results')
#
#     fig = model.plot_data(feature='phase')
#     plt.show(block=False)
#
#     fig = model.plot_traces()
#     # fig.savefig('./docs/tutorials/figures/ex1_traces.png', dpi=144, bbox_inches='tight')
#     plt.show(block=False)
#
#     fig = model.plot_histograms(chain)
#     # fig.savefig('./figures/histograms.png', dpi=144, bbox_inches='tight')
#     plt.close()
#
#     fig = model.plot_fit(chain)
#     # fig.savefig('./figures/fitted.png', dpi=144, bbox_inches='tight')
#     plt.show(block=False)
#
#     try:
#         fig = model.plot_corner(chain)
#         plt.close()
#         # fig.savefig('./figures/corner.png', dpi=144, bbox_inches='tight')
#     except ImportError:
#         warnings.warn('The `corner` package was not found. Install it with '
#                       '`conda install corner`')
#
#     print('All tests passed. Press ctrl+C or close figure windows to exit.')
#     plt.show()
