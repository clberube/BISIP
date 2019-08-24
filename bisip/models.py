# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 12:05:22 2015

@author:    charleslberube@gmail.com
            École Polytechnique de Montréal

The MIT License (MIT)

Copyright (c) 2016 Charles L. Bérubé

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the 'Software'), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED 'AS IS', WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

https://opensource.org/licenses/MIT
https://github.com/clberube/bisip

This python module may be used to import SIP data, run MCMC inversion and
return the results.
"""

from __future__ import division
from __future__ import print_function

#==============================================================================
# Import PyMC, Numpy, and Cython extension
from builtins import range
from past.utils import old_div
import pymc
import numpy as np
from bisip.cython_funcs import ColeCole_cyth1, Dias_cyth, Decomp_cyth, Shin_cyth
# Imports to save things
from os import path, makedirs
from os import getcwd
from datetime import datetime
from scipy.signal import argrelextrema

from bisip import invResults as iR
from bisip.utils import format_results, get_data
from bisip.utils import split_filepath, get_model_type
from bisip.utils import var_depth, flatten

try:
    import lib_dd.decomposition.ccd_single as ccd_single
    import lib_dd.config.cfg_single as cfg_single
    print('\nCCDtools available')
except:
    pass

import matplotlib as mpl
mpl.rc_file_defaults()


def run_MCMC(function, mc_p, save_traces=False, save_where=None):
    # Function to run MCMC simulation on selected model
    # Arguments: model <function>, mcmc parameters <dict>,traces path <string>
    print('\nMCMC parameters:\n', mc_p)
    if save_traces:
        # If path doesn't exist, create it
        if not path.exists(save_where):
            makedirs(save_where)
        MDL = pymc.MCMC(function, db='txt', dbname=save_where)
    else:
        MDL = pymc.MCMC(function, db='ram', dbname=save_where)

    if mc_p['adaptive']:
        if mc_p['verbose']:
            mc_p['verbose'] = 1
        MDL.use_step_method(pymc.AdaptiveMetropolis,
                            MDL.stochastics,
                            delay=mc_p['cov_delay'],
                            interval=mc_p['cov_inter'],
                            shrink_if_necessary=True,
                            verbose=mc_p['verbose'])

    else:
        for stoc in MDL.stochastics:
            MDL.use_step_method(pymc.Metropolis,
                                stoc,
                                proposal_distribution='Normal',
                                scale=mc_p['prop_scale'],
                                verbose=mc_p['verbose'])

    for i in range(1, mc_p['nb_chain']+1):
        print('\nChain #%d/%d' % (i, mc_p['nb_chain']))
        MDL.sample(mc_p['nb_iter'],
                   mc_p['nb_burn'],
                   mc_p['thin'],
                   tune_interval=mc_p['tune_inter'],
                   tune_throughout=False)
    return MDL

class mcmcinv(object):
    """
    List of models:
    Use mcmcinv('ColeCole'
                'Dias'
                'PDecomp'
                'Shin'
                'CCDtools'
                )
    """

    # Default MCMC parameters:
    default_mcmc = {'adaptive': True,
                    'nb_chain': 1,
                    'nb_iter': 10000,
                    'nb_burn': 8000,
                    'thin': 1,
                    'tune_inter': 500,
                    'prop_scale': 1.0,
                    'verbose': False,
                    'cov_inter': 1000,
                    'cov_delay': 1000,
                    }

    # Define some attributes of mcmcinv
    print_results = iR.print_resul
    plot_fit = iR.plot_fit
    plot_histograms = iR.plot_histo
    plot_traces = iR.plot_traces
    save_results = iR.save_resul
    save_csv_traces = iR.save_csv_traces
    merge_results = iR.merge_results
    plot_log_likelihood = iR.plot_logp
    plot_model_deviance = iR.plot_deviance
    plot_data = iR.plot_data
    plot_rtd = iR.plot_rtd
    plot_autocorrelation = iR.plot_autocorr
    plot_summary = iR.plot_summary
    plot_hexbin = iR.plot_hexbin
    plot_KDE = iR.plot_KDE

    def __init__(self, model, filepath, mcmc=default_mcmc, headers=1,
                 ph_units='mrad', cc_modes=2, decomp_poly=4, c_exp=1.0,
                 log_min_tau=-3, guess_noise=False, keep_traces=False,
                 ccdt_priors='auto', ccdt_cfg=None):
        """
        Call with minimal arguments:
        sol = mcmcinv('ColeCole', '/Documents/DataFiles/DATA.dat')

        Call with all optional arguments:
        sol = mcmcinv( model='ColeCole', filepath='/Documents/DataFiles/DATA.dat',
                 mcmc=mcmc_params, headers=1, ph_units='mrad', cc_modes=2,
                 debye_poly=4, c_exp = 1.0, keep_traces=False)
        """

        self.model = model
        self.filepath = filepath
        self.mcmc = mcmc
        self.headers = headers
        self.ph_units = ph_units
        self.cc_modes = cc_modes
        self.decomp_poly = decomp_poly
        self.c_exp = c_exp
        self.log_min_tau = log_min_tau
        self.guess_noise = guess_noise
        self.keep_traces = keep_traces
        self.ccd_priors = ccdt_priors
        self.ccdtools_config = ccdt_cfg
        self.ccdt_last_it = None
        self.filename = split_filepath(self.filepath)

        if model == 'CCD':
            if self.ccd_priors == 'auto':
                self.ccd_priors = self.get_ccd_priors(config=self.ccdtools_config)[0]
            self.ccdt_last_it = self.get_ccd_priors(config=self.ccdtools_config)[1]
            print('\nUpdated CCD priors with new data')

        # self.start()

    def get_ccd_priors(self, config=None):
        data = get_data(self.filepath, self.headers, self.ph_units)
        data_ccdtools = np.hstack((data['amp'][::-1], 1000*data['pha'][::-1]))
        freq_ccdtools = data['freq'][::-1]
        if config is None:
            config = cfg_single.cfg_single()
            config['fixed_lambda'] = 20
            config['norm'] = 10

        config['frequency_file'] = freq_ccdtools
        config['data_file'] = data_ccdtools
        # generate a ccd object
        ccd_obj = ccd_single.ccd_single(config)

        # commence with the actual fitting
        ccd_obj.fit_data()

        # extract the last iteration
        ccdt_last_it = ccd_obj.results[-1].iterations[-1]

        # Make a dictionary with what we learn from CCDtools inversion
        priors = {}
        priors['R0'] = ccdt_last_it.stat_pars['rho0'][0]
        priors['tau'] = ccdt_last_it.Data.obj.tau
        priors['log_tau'] = np.log10(ccdt_last_it.Data.obj.tau)
        priors['m'] = 10**ccdt_last_it.m[1:]
        priors['log_m'] = ccdt_last_it.m[1:]
        return priors, ccdt_last_it

    def ColeColeModel(self, cc_modes):

        """Cole-Cole Bayesian Model
        """
        # Initial guesses
        p0 = {'R0': 1.0,
              'm': None,
              'log_tau': None,
              'c': None,
              }

        # Stochastic variables
        R0 = pymc.Uniform('R0', lower=0.7, upper=1.3, value=p0['R0'])
        m = pymc.Uniform('m', lower=0.0, upper=1.0, value=p0['m'],
                         size=cc_modes)
        log_tau = pymc.Uniform('log_tau', lower=-7.0, upper=4.0,
                               value=p0['log_tau'], size=cc_modes)
        c = pymc.Uniform('c', lower=0.0, upper=1.0, value=p0['c'],
                         size=cc_modes)

        # Deterministic variables
        @pymc.deterministic()
        def zmod(R0=R0, m=m, lt=log_tau, c=c):
            return ColeCole_cyth1(self.w, R0, m, lt, c)

        @pymc.deterministic()
        def NRMSE_r(zmod=zmod, data=self.data['zn']):
            return np.sqrt(np.mean((zmod[0] - data[0])**2))/abs(max(data[0])-min(data[0]))

        @pymc.deterministic()
        def NRMSE_i(zmod=zmod, data=self.data['zn']):
            return np.sqrt(np.mean((zmod[1] - data[1])**2))/abs(max(data[1])-min(data[1]))

        # Likelihood
        obs = pymc.Normal('obs', mu=zmod,
                          tau=old_div(1.0, (self.data['zn_err']**2)),
                          value=self.data['zn'], size=(2, len(self.w)),
                          observed=True)

        return locals()

    def ShinModel(self):
        """Shin Bayesian Model"""
        # Initial guesses
        p0 = {'R': [0.5, 0.5],
              'log_Q': [0, -4],
              'n': [0.5, 0.5],
              'log_tau': None,
              'm': None,
              }
        # Stochastics
        R = pymc.Uniform('R', lower=0.0, upper=1.0, value=p0['R'], size=2)
        log_Q = pymc.Uniform('log_Q', lower=-7, upper=2,
                             value=p0['log_Q'], size=2)
        n = pymc.Uniform('n', lower=0.0, upper=1.0, value=p0['n'], size=2)
        # Deterministics
        @pymc.deterministic(plot=False)
        def zmod(R=R, log_Q=log_Q, n=n):
            return Shin_cyth(self.w, R, log_Q, n)

        @pymc.deterministic(plot=False)
        def log_tau(R=R, log_Q=log_Q, n=n):
            return np.log10((R*(10**log_Q))**(old_div(1., n)))

        @pymc.deterministic(plot=False)
        def R0(R=R):
            return R[0]+R[1]

        @pymc.deterministic(plot=False)
        def m(R=R):
            return self.seigle_m*(old_div(max(R), (max(R) + min(R))))

        @pymc.deterministic(plot=False)
        def NRMSE_r(zmod=zmod, data=self.data['zn']):
            return np.sqrt(np.mean((zmod[0] - data[0])**2))/abs(max(data[0])-min(data[0]))

        @pymc.deterministic(plot=False)
        def NRMSE_i(zmod=zmod, data=self.data['zn']):
            return np.sqrt(np.mean((zmod[1] - data[1])**2))/abs(max(data[1])-min(data[1]))

        #Likelihood
        obs = pymc.Normal('obs', mu=zmod,
                          tau=old_div(1.0, (self.data['zn_err']**2)),
                          value=self.data['zn'], size=(2, len(self.w)),
                          observed=True)

        return locals()

    def DiasModel(self):
        """Dias Bayesian Model
        """
        # Initial guesses
        p0 = {'R0': 1.0,
              'm': self.seigle_m,
              'log_tau': None,
              'eta': None,
              'delta': None,
              }
        # Stochastics
        R0 = pymc.Uniform('R0', lower=0.9, upper=1.1, value=1)
        m = pymc.Uniform('m', lower=0.0, upper=1.0, value=p0['m'])
        log_tau = pymc.Uniform('log_tau', lower=-7.0, upper=0.0,
                               value=p0['log_tau'])
        eta = pymc.Uniform('eta', lower=0.0, upper=50.0, value=p0['eta'])
        delta = pymc.Uniform('delta', lower=0.0, upper=1.0,
                             value=p0['delta'])

        # Deterministics
        @pymc.deterministic(plot=False)
        def zmod(R0=R0, m=m, lt=log_tau, eta=eta, delta=delta):
            return Dias_cyth(self.w, R0, m, lt, eta, delta)

        # Likelihood
        obs = pymc.Normal('obs', mu=zmod,
                          tau=old_div(1.0, (self.data['zn_err']**2)),
                          value=self.data['zn'], size=(2, len(self.w)),
                          observed=True)

        @pymc.deterministic(plot=False)
        def NRMSE_r(zmod=zmod, data=self.data['zn']):
            return np.sqrt(np.mean((zmod[0] - data[0])**2))/abs(max(data[0])-min(data[0]))

        @pymc.deterministic(plot=False)
        def NRMSE_i(zmod=zmod, data=self.data['zn']):
            return np.sqrt(np.mean((zmod[1] - data[1])**2))/abs(max(data[1])-min(data[1]))

        return locals()

    def stoCCD(self, c_exp, ccd_priors):
        # Stochastic variables (noise on CCDtools output)
        # The only assumption we make is that the RTD noise is
        # assumed to be equal to 0 and below 20% with 1 standard deviation
        noise_tau = pymc.Normal('log_noise_tau', mu=0, tau=1/(0.2**2))
        noise_m = pymc.Normal('log_noise_m', mu=0, tau=1/(0.2**2))
        noise_rho = pymc.Normal('log_noise_rho', mu=0, tau=1/(0.2**2))

        # Deterministic variables of CCD
        @pymc.deterministic(plot=False)
        def log_m_i(logm=ccd_priors['log_m'], dm=noise_m):
            # log chargeability array
            return logm + dm

        @pymc.deterministic(plot=False)
        def log_tau_i(logt=ccd_priors['log_tau'], dt=noise_tau):
            # log tau array
            return logt + dt

        @pymc.deterministic(plot=False)
        def R0(R=ccd_priors['R0'], dR=noise_rho):
            # DC resistivity (normalized)
            return R + dR

        @pymc.deterministic(plot=False)
        def cond(log_tau=log_tau_i):
            # Condition on log_tau to compute integrating parameters
            log_tau_min = np.log10(1./self.w.max())
            log_tau_max = np.log10(1./self.w.min())
            return (log_tau >= log_tau_min) & (log_tau <= log_tau_max)

        @pymc.deterministic(plot=False)
        def log_total_m(m=10**log_m_i[cond]):
            # Total chargeability
            return np.log10(np.nansum(m))

        @pymc.deterministic(plot=False)
        def log_half_tau(m_i=10**log_m_i[cond], log_tau=log_tau_i[cond]):
            # Tau 50
            return log_tau[np.where(np.cumsum(m_i)/np.nansum(m_i) > 0.5)[0][0]]

        @pymc.deterministic(plot=False)
        def log_U_tau(m_i=10**log_m_i[cond], log_tau=log_tau_i[cond]):
            tau_60 = log_tau[np.where(np.cumsum(m_i)/np.nansum(m_i) > 0.6)[0][0]]
            tau_10 = log_tau[np.where(np.cumsum(m_i)/np.nansum(m_i) > 0.1)[0][0]]
            return np.log10(10**tau_60 / 10**tau_10)

        @pymc.deterministic(plot=False)
        def log_peak_tau(m_i=log_m_i, log_tau=log_tau_i):
            # Tau peaks
            peak_cond = argrelextrema(m_i, np.greater)
            return np.squeeze(log_tau[peak_cond])

        @pymc.deterministic(plot=False)
        def log_peak_m(log_m=log_m_i):
            peak_cond = argrelextrema(log_m, np.greater)
            # Peak chargeability
            return np.squeeze(log_m[peak_cond])

        @pymc.deterministic(plot=False)
        def log_mean_tau(m_i=10**log_m_i[cond], log_tau=log_tau_i[cond]):
            # Tau logarithmic average
            return np.log10(np.exp(np.nansum(m_i*np.log(10**log_tau))/np.nansum(m_i)))

        @pymc.deterministic(plot=False)
        def zmod(R0=R0, m=10**log_m_i, tau=10**log_tau_i):
            Z = R0 * (1 - np.sum(m*(1 - 1.0/(1 + ((1j*self.w[:, np.newaxis]*tau)**c_exp))), axis=1))
            return np.array([Z.real, Z.imag])

        @pymc.deterministic(plot=False)
        def NRMSE_r(zmod=zmod, data=self.data['zn']):
            return np.sqrt(np.mean((zmod[0] - data[0])**2))/abs(max(data[0])-min(data[0]))

        @pymc.deterministic(plot=False)
        def NRMSE_i(zmod=zmod, data=self.data['zn']):
            return np.sqrt(np.mean((zmod[1] - data[1])**2))/abs(max(data[1])-min(data[1]))

        # Likelihood function
        obs = pymc.Normal('obs', mu=zmod,
                          tau=1./(2*self.data['zn_err']**2),
                          value=self.data['zn'], size=(2, len(self.w)),
                          observed=True)

        return locals()

    def PolyDecompModel(self, decomp_poly, c_exp, ccd_priors):
        """Debye, Warburg, Cole-Cole decomposition Bayesian Model
        """
        # Initial guesses
        p0 = {'R0': 1.0,
              'a': None,
              'log_tau_hi': -5.0,
              'm_hi': 0.5,
              'TotalM': None,
              'log_MeanTau': None,
              'U': None,
              }
        # Stochastics
        R0 = pymc.Uniform('R0', lower=0.7, upper=1.3, value=p0['R0'])
        a = pymc.Normal('a', mu=0, tau=1./(0.01**2), value=p0['a'],
                        size=decomp_poly+1)
        if self.guess_noise:
            noise_r = pymc.Uniform('noise_real', lower=0., upper=1.)
            noise_i = pymc.Uniform('noise_imag', lower=0., upper=1.)

        @pymc.deterministic(plot=False)
        def zmod(R0=R0, a=a):
            return Decomp_cyth(self.w, self.tau_10, self.log_taus, c_exp, R0, a)

        @pymc.deterministic(plot=False)
        def m_i(a=a):
            return np.sum((a*self.log_taus.T).T, axis=0)

        @pymc.deterministic(plot=False)
        def total_m(m=m_i[self.cond]):
            return np.nansum(m)

        @pymc.deterministic(plot=False)
        def log_half_tau(m=m_i[self.cond], log_tau=self.log_tau[self.cond]):
            # Tau 50
            return self.log_tau[np.where(np.cumsum(m)/np.nansum(m) > 0.5)[0][0]]

        @pymc.deterministic(plot=False)
        def log_mean_tau(m=m_i[self.cond], log_tau=self.log_tau[self.cond]):
            return np.log10(np.exp(old_div(np.sum(m*np.log(10**log_tau)), np.sum(m))))

        @pymc.deterministic(plot=False)
        def log_U_tau(m=m_i[self.cond], log_tau=self.log_tau[self.cond]):
            tau_60 = log_tau[np.where(np.cumsum(m)/np.nansum(m) > 0.6)[0][0]]
            tau_10 = log_tau[np.where(np.cumsum(m)/np.nansum(m) > 0.1)[0][0]]
            return np.log10(10**tau_60 / 10**tau_10)

        @pymc.deterministic(plot=False)
        def NRMSE_r(zmod=zmod, data=self.data['zn']):
            return np.sqrt(np.mean((zmod[0] - data[0])**2))/abs(max(data[0])-min(data[0]))

        @pymc.deterministic(plot=False)
        def NRMSE_i(zmod=zmod, data=self.data['zn']):
            return np.sqrt(np.mean((zmod[1] - data[1])**2))/abs(max(data[1])-min(data[1]))

        if self.guess_noise:
            obs_r = pymc.Normal('obs_r', mu=zmod[0], tau=1./((noise_r)**2),
                                value=self.data['zn'][0], size=len(self.w),
                                observed=True)
            obs_i = pymc.Normal('obs_i', mu=zmod[1], tau=1./((noise_i)**2),
                                value=self.data['zn'][1], size=len(self.w),
                                observed=True)
        else:
            obs = pymc.Normal('obs', mu=zmod,
                              tau=1./(self.data['zn_err']**2),
                              value=self.data['zn'],
                              size=(2, len(self.w)), observed=True)

        return locals()

    def fit(self):
        """
        Main section
        """

        # Importing data
        self.data = get_data(self.filepath, self.headers, self.ph_units)

        data_ccd = np.hstack((self.data['amp'][::-1], 1000*self.data['pha'][::-1]))
        frequencies_ccd = self.data['freq'][::-1]

        # generate a ccd object
        if self.model == 'CCD':
            self.obj = ccd_single.ccd_single(cfg_single.cfg_single())
            self.obj.config['frequency_file'] = frequencies_ccd
            self.obj.config['data_file'] = data_ccd

        if (self.data['pha_err'] == 0).all():
            self.guess_noise = True
        # Estimating Seigel chargeability
        self.seigle_m = (old_div((self.data['amp'][-1] - self.data['amp'][0]), self.data['amp'][-1]))
        w = 2*np.pi*self.data['freq'] # Frequencies measured in rad/s
        # Relaxation times associated with the measured frequencies (Debye decomposition only)
        self.w = w

        if self.model == 'PDecomp':
            log_tau = np.linspace(np.floor(min(np.log10(old_div(1.0,w)))-1), np.floor(max(np.log10(old_div(1.0,w)))+1), 50)
            cond = (log_tau >= min(log_tau)+1)&(log_tau <= max(log_tau)-1)
            # Polynomial approximation for the RTD
            log_taus = np.array([log_tau**i for i in list(reversed(range(0,self.decomp_poly+1)))])
            tau_10 = 10**log_tau  # Accelerates sampling
            self.data['tau'] = tau_10  # Put relaxation times in data dictionary
            self.log_taus = log_taus
            self.log_tau = log_tau
            self.tau_10 = tau_10
            self.cond = cond

        # Time and date (for saving traces)
        sample_name = self.filepath.replace('\\', '/').split('/')[-1].split('.')[0]
        working_path = getcwd().replace('\\', '/')+'/'
        now = datetime.now()
        save_time = now.strftime('%Y%m%d_%H%M%S')
        save_date = now.strftime('%Y%m%d')
        out_path = '%s/Txt traces/%s/%s/%s-%s-%s/' % (working_path, save_date,
                                                      sample_name, self.model,
                                                      sample_name, save_time)

        """
        #==========================================================================
        Call to run_MCMC function
        #==========================================================================
        """
        # 'ColeCole', 'Dias', 'Debye' or 'Shin'
        sim_dict = {'ColeCole': {'func': self.ColeColeModel, 'args': [self.cc_modes]},
                    'Dias': {'func': self.DiasModel, 'args': []},
                    'PDecomp': {'func': self.PolyDecompModel, 'args': [self.decomp_poly, self.c_exp, self.ccd_priors]},
                    'Shin': {'func': self.ShinModel, 'args': []},
    #                'Custom':   {'func': YourModel,     'args': [opt_args]   },
                    'CCD': {'func': self.stoCCD, 'args': [self.c_exp, self.ccd_priors]},
                    }
        simulation = sim_dict[self.model]  # Pick entries for the selected model
        self.MDL = run_MCMC(simulation['func'](*simulation['args']), self.mcmc, save_traces=self.keep_traces, save_where=out_path)  # Run MCMC simulation with selected model and arguments

        """
        #==========================================================================
        Results
        #==========================================================================
        """

        self.pm = format_results(self.MDL, self.data['Z_max'])  # Format output
        zmodstats = self.MDL.stats(chain=-1)['zmod']  # Take last chain
        print(zmodstats)
        zn_avg = zmodstats['mean']
        zn_l95 = zmodstats['95% HPD interval'][0]
        zn_u95 = zmodstats['95% HPD interval'][1]
        avg = self.data['Z_max']*(zn_avg[0] + 1j*zn_avg[1])  # (In complex notation, de-normalized)
        l95 = self.data['Z_max']*(zn_l95[0] + 1j*zn_l95[1])  # (In complex notation, de-normalized)
        u95 = self.data['Z_max']*(zn_u95[0] + 1j*zn_u95[1])  # (In complex notation, de-normalized)
        self.fit = {'best': avg, 'lo95': l95, 'up95': u95}  # Best fit dict with 95% HDP
        self.model_type = {'log_min_tau': self.log_min_tau,
                           'c_exp': self.c_exp,
                           'decomp_polyn': self.decomp_poly,
                           'cc_modes': self.cc_modes}
        self.model_type_str = get_model_type(self)
        self.var_dict = dict([(x.__name__,x) for x in self.MDL.deterministics] + [(x.__name__,x) for x in self.MDL.stochastics])

        # Get names of parameters for save file
        pm_names = [x for x in sorted(self.var_dict.keys())]
        # Get all stochastic and deterministic variables
        trl = [self.var_dict[x] for x in pm_names]
        # Concatenate all traces in 1 matrix
        trace_mat = np.hstack([t.trace().reshape(-1, var_depth(t)) for t in trl])
        # Get numbers for each subheader
        num_names = [var_depth(v) for v in trl]
        # Make list of headers
        headers = flatten([['%s%d'%(pm_names[p], x+1) for x in range(num_names[p])] if num_names[p] > 1 else [pm_names[p]] for p in range(len(pm_names))])

        self.trace_dict = {k: t for k, t in zip(headers, trace_mat.T)}

        # End of inversion
