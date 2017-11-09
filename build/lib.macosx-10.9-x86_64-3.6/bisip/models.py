# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 12:05:22 2015

@author:    charleslberube@gmail.com
            École Polytechnique de Montréal

The MIT License (MIT)

Copyright (c) 2016 Charles L. Bérubé

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
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

It is imported as:

Call with minimal arguments:

sol = mcmcinv('ColeCole', '/Documents/DataFiles/DATA.dat')

Call with all optional arguments:

sol = mcmcinv( model='ColeCole', filename='/Documents/DataFiles/DATA.dat',
                 mcmc=mcmc_params, headers=1, ph_units='mrad', cc_modes=2,
                 debye_poly=4, c_exp = 1.0, keep_traces=False)
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
import bisip.invResults as iR
from bisip.utils import format_results, get_data
import lib_dd.decomposition.ccd_single as ccd_single
import lib_dd.config.cfg_single as cfg_single

#==============================================================================
# Function to run MCMC simulation on selected model
# Arguments: model <function>, mcmc parameters <dict>,traces path <string>
def run_MCMC(function, mc_p, save_traces=False, save_where=None):
    print("\nMCMC parameters:\n", mc_p)
    if save_traces:
        # If path doesn't exist, create it
        if not path.exists(save_where): makedirs(save_where)
        MDL = pymc.MCMC(function, db='txt',
                        dbname=save_where)
    else:
        MDL = pymc.MCMC(function, db='ram',
                        dbname=save_where)

    if mc_p["adaptive"]:
        if mc_p['verbose']:
            mc_p['verbose'] = 1
        MDL.use_step_method(pymc.AdaptiveMetropolis, MDL.stochastics, delay=mc_p["cov_delay"], interval=mc_p['cov_inter'], shrink_if_necessary=True, verbose=mc_p['verbose'])

    else:
        for stoc in MDL.stochastics:
            MDL.use_step_method(pymc.Metropolis, stoc,
                                proposal_distribution='Normal',
                                scale=mc_p['prop_scale'], verbose=mc_p['verbose'])

    for i in range(1, mc_p['nb_chain']+1):
        print('\nChain #%d/%d'%(i, mc_p['nb_chain']))
        MDL.sample(mc_p['nb_iter'], mc_p['nb_burn'], mc_p['thin'], tune_interval=mc_p['tune_inter'], tune_throughout=False)
    return MDL

class mcmcinv(object):
    """
    List of models:
    Use mcmcinv("ColeCole"
                "Dias"
                "PDecomp"
                "Shin"
                "CCDtools"
                )
    """
    
    # Default MCMC parameters:
    default_mcmc = {"adaptive"  : True,
                   "nb_chain"   : 1,
                   "nb_iter"    : 10000,
                   "nb_burn"    : 8000,
                   "thin"       : 1,
                   "tune_inter" : 500,
                   "prop_scale" : 1.0,
                   "verbose"    : False,
                   "cov_inter"  : 1000,
                   "cov_delay"  : 1000,
                    }
    
    print_results = iR.print_resul
    plot_fit = iR.plot_fit
    plot_histograms = iR.plot_histo
    plot_traces = iR.plot_traces
    save_results = iR.save_resul
    merge_results = iR.merge_results
    plot_log_likelihood = iR.plot_logp
    plot_model_deviance = iR.plot_deviance
    plot_data = iR.plot_data
    plot_rtd = iR.plot_rtd
    plot_autocorrelation = iR.plot_autocorr
    plot_summary = iR.plot_summary
    plot_hexbin = iR.plot_hexbin
    plot_KDE = iR.plot_KDE
    get_model_type = iR.get_model_type
    
    

    
    
    
    
    def __init__(self, model, filename, mcmc=default_mcmc, headers=1,
                   ph_units="mrad", cc_modes=2, decomp_poly=4, c_exp=1.0, 
                   log_min_tau=-3, guess_noise=False, keep_traces=False, 
                   ccdt_priors='auto', ccdt_cfg=None):
        
        self.model = model
        self.filename = filename 
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
        if model == "CCD":
            if self.ccd_priors == 'auto':
                self.ccd_priors = self.get_ccd_priors(config=self.ccdtools_config)
                print("\nUpdated CCD priors with new data")
        self.start()
        

                
#    def print_resul(self):
#    #==============================================================================
#        # Impression des résultats
#        pm, model, filename = self.pm, self.model, self.filename
#        print('\n\nInversion success!')
#        print('Name of file:', filename)
#        print('Model used:', model)
#        e_keys = sorted([s for s in list(pm.keys()) if "_std" in s])
#        v_keys = [e.replace("_std", "") for e in e_keys]
#        labels = ["{:<8}".format(x+":") for x in v_keys]
#        np.set_printoptions(formatter={'float': lambda x: format(x, '6.3E')})
#        for l, v, e in zip(labels, v_keys, e_keys):
#            print(l, pm[v], '+/-', pm[e], np.char.mod('(%.2f%%)',abs(100*pm[e]/pm[v])))
    
    def get_ccd_priors(self, config=None):
        data = get_data(self.filename, self.headers, self.ph_units)
#        print(data['pha'][0])
        data_ccdtools = np.hstack((data['amp'][::-1], 1000*data['pha'][::-1]))
        freq_ccdtools = data['freq'][::-1]
#        print(data_ccdtools[0],data_ccdtools[5])
        # set options using this dict-like object

#        if config == None:
#            config.update(cfg_single.cfg_single())
#            config['fixed_lambda'] = 10
#            config['norm'] = 10
#            print("\nNo CCDtools config passed, using default")
            
#        config.update({'frequency_file': freq_ccdtools})
#        config.update({'data_file': data_ccdtools})          
#        print(config['data_file'][0])
    

#        print(config['data_file'][-1])
#        print(config['data_file'][5])
#        print(config['frequency_file'][0])


#        print(config['data_file'][-1])
        if config == None:
            config = cfg_single.cfg_single()
            config['fixed_lambda'] = 10
            config['norm'] = 10
                        
        config['frequency_file'] = freq_ccdtools
        config['data_file'] = data_ccdtools        
        # generate a ccd object
        ccd_obj = ccd_single.ccd_single(config)
        
        # commence with the actual fitting
        ccd_obj.fit_data()
        
        # extract the last iteration
        last_it = ccd_obj.results[-1].iterations[-1]
        
        # Make a dictionary with what we learn from CCDtools inversion
        priors = {}
        priors['R0'] = last_it.stat_pars['rho0'][0]
        priors['tau'] = last_it.Data.obj.tau
        priors['log_tau'] = np.log10(last_it.Data.obj.tau)
        priors['m'] = 10**last_it.m[1:]
        priors['log_m'] = last_it.m[1:]
        print(last_it.stat_pars['rho0'])
#        print(priors['log_m'][0])
        return priors 
    
    #==============================================================================
    # Main inversion function.
    def start(self):
    #==============================================================================
        """Cole-Cole Bayesian Model"""
    #==============================================================================
        def ColeColeModel(cc_modes):
            # Initial guesses
            p0 = {'R0'       : 1.0,
                  'm'        : None,
                  'log_tau'  : None,
                  'c'        : None,
                  }
            # Stochastics
            R0 = pymc.Uniform('R0', lower=0.7, upper=1.3 , value=p0["R0"])
            m = pymc.Uniform('m', lower=0.0, upper=1.0, value=p0["m"], size=cc_modes)
            log_tau = pymc.Uniform('log_tau', lower=-7.0, upper=4.0, value=p0['log_tau'], size=cc_modes)
            c = pymc.Uniform('c', lower=0.0, upper=1.0, value=p0['c'], size=cc_modes)
            # Deterministics
            @pymc.deterministic(plot=False)
            def zmod(cc_modes=cc_modes, R0=R0, m=m, lt=log_tau, c=c):
                return ColeCole_cyth1(w, R0, m, lt, c)
            # Likelihood
            obs = pymc.Normal('obs', mu=zmod, tau=old_div(1.0,(self.data["zn_err"]**2)), value=self.data["zn"], size=(2,len(w)), observed=True)
            return locals()
    
    #==============================================================================
        """Shin Bayesian Model"""
    #==============================================================================
        def ShinModel():
            # Initial guesses
            p0 = {'R'      : [0.5, 0.5],
                  'log_Q'  : [0,-4],
                  'n'      : [0.5, 0.5],
                  'log_tau': None,
                  'm'      : None,
                  }
            # Stochastics
            R = pymc.Uniform('R', lower=0.0, upper=1.0, value=p0["R"], size=2)
            log_Q = pymc.Uniform('log_Q', lower=-7, upper=2, value=p0["log_Q"], size=2)
            n = pymc.Uniform('n', lower=0.0, upper=1.0, value=p0["n"], size=2)
            # Deterministics
            @pymc.deterministic(plot=False)
            def zmod(R=R, log_Q=log_Q, n=n):
                return Shin_cyth(w, R, log_Q, n)
            @pymc.deterministic(plot=False)
            def log_tau(R=R, log_Q=log_Q, n=n):
                return np.log10((R*(10**log_Q))**(old_div(1.,n)))
            @pymc.deterministic(plot=False)
            def R0(R=R):
                return R[0]+R[1]
            @pymc.deterministic(plot=False)
            def m(R=R):
                return seigle_m*( old_div(max(R), (max(R) + min(R))))
            #Likelihood
            obs = pymc.Normal('obs', mu=zmod, tau=old_div(1.0,(self.data["zn_err"]**2)), value=self.data["zn"], size = (2,len(w)), observed=True)
            return locals()
    
    #==============================================================================
        """Dias Bayesian Model"""
    #==============================================================================
        def DiasModel():
            # Initial guesses
            p0 = {'R0'     :  1.0,
                  'm'      :  seigle_m,
                  'log_tau':  None,
                  'eta'    :  None,
                  'delta'  :  None,
                  }
            # Stochastics
            R0 = pymc.Uniform('R0', lower=0.9, upper=1.1 , value=1)
            m = pymc.Uniform('m', lower=0.0, upper=1.0, value=p0['m'])
            log_tau = pymc.Uniform('log_tau', lower=-7.0, upper=0.0, value=p0['log_tau'])
            eta = pymc.Uniform('eta', lower=0.0, upper=50.0, value=p0['eta'])
            delta = pymc.Uniform('delta', lower=0.0, upper=1.0, value=p0['delta'])
            # Deterministics
            @pymc.deterministic(plot=False)
            def zmod(R0=R0, m=m, lt=log_tau, eta=eta, delta=delta):
                return Dias_cyth(w, R0, m, lt, eta, delta)
            # Likelihood
            obs = pymc.Normal('obs', mu=zmod, tau=old_div(1.0,(self.data["zn_err"]**2)), value=self.data["zn"], size = (2,len(w)), observed=True)
            return locals()
    
    
        def stoCCD(c_exp, ccd_priors):
            # Stochastic variables (noise on CCDtools output)
            # The only assumption we make is that the RTD noise is below 10%
            noise_rho = pymc.Uniform('noise_rho', lower=-0.1, upper=0.1)
            noise_tau = pymc.Uniform('log_noise_tau', lower=-0.1, upper=0.1)
            noise_m = pymc.Uniform('log_noise_m', lower=-0.1, upper=0.1)
            
            # Deterministic variables of CCD
            @pymc.deterministic(plot=False) 
            def log_m_i(logm=ccd_priors['log_m'], dm=noise_m):
                # Chargeability array
                return logm + dm
            @pymc.deterministic(plot=False) 
            def log_tau_i(logt=ccd_priors['log_tau'], dt=noise_tau):
                # Tau logarithmic array
                return logt + dt
            @pymc.deterministic(plot=False) 
            def R0(R=ccd_priors['R0'], dR=noise_rho):
                # DC resistivity (normalized)
                return R + dR
            @pymc.deterministic(plot=False) 
            def cond(log_tau = log_tau_i):
                # Condition on log_tau to compute integrating parameters
                return (log_tau >= min(log_tau)+1)&(log_tau <= max(log_tau)-1)
            @pymc.deterministic(plot=False)
            def total_m(m=10**log_m_i[cond]):
                # Total chargeability
                return np.sum(m)
            @pymc.deterministic(plot=False)
            def log_half_tau(m_i=10**log_m_i[cond], log_tau=log_tau_i[cond]):
                # Tau 50
                return log_tau[np.where(np.cumsum(m_i)/np.sum(m_i) > 0.5)[0][0]]
            @pymc.deterministic(plot=False)
            def log_peak_tau(m_i=log_m_i, log_tau=log_tau_i):
                # Tau peaks
                peak_cond = np.r_[True, m_i[1:] > m_i[:-1]] & np.r_[m_i[:-1] > m_i[1:], True]
                return log_tau[peak_cond]
            @pymc.deterministic(plot=False)
            def log_mean_tau(m_i=10**log_m_i[cond], log_tau=log_tau_i[cond]):
                # Tau logarithmic average 
                return np.log10(np.exp(old_div(np.sum(m_i*np.log(10**log_tau)),np.sum(m_i))))
            @pymc.deterministic(plot=False)
            def zmod(R0=R0, m=10**log_m_i, tau=10**log_tau_i):
                Z = R0 * (1 - np.sum(m*(1 - 1.0/(1 + ((1j*w[:,np.newaxis]*tau)**c_exp))), axis=1))
                return np.array([Z.real, Z.imag])
            
            # Likelihood function
            obs = pymc.Normal('obs', mu=zmod, tau=1./(2*self.data["zn_err"]**2), value=self.data["zn"], size = (2, len(w)), observed=True)
            return locals()
    
    #==============================================================================
        """Debye, Warburg, Cole-Cole decomposition Bayesian Model"""
    #==============================================================================
        def PolyDecompModel(decomp_poly, c_exp, ccd_priors):
            # Initial guesses
            p0 = {'R0'         : 1.0,
                  'a'          : None,
    #              'a'          : ([0.01, -0.01, -0.01, 0.001, 0.001]+[0.0]*(decomp_poly-4))[:(decomp_poly+1)],
                  'a_mu' : np.array([0.00590622364129, -0.00259869937567, -0.00080727429007, 0.00051369743841, 0.000176048226508]),
                  'a_sd' : np.array([0.00448686724083, 0.00354717249566, 0.00153254695967, 0.00109002742145, 0.000189386869372]),
                  'log_tau_hi' : -5.0,
                  'm_hi'       : 0.5,
                  'TotalM'     : None,
                  'log_MeanTau': None,
                  'U'          : None,
                  }
            # Stochastics
            R0 = pymc.Uniform('R0', lower=0.7, upper=1.3, value=p0['R0'])
#            R0 = pymc.Normal('R0', mu=0.989222579813, tau=1./(0.0630422467962**2))
#            R0 = pymc.Normal('R0', mu=ccd_priors['R0'], tau=1./(1e-10**2))

    #        m_hi = pymc.Uniform('m_hi', lower=0.0, upper=1.0, value=p0['m_hi'])
    #        log_tau_hi = pymc.Uniform('log_tau_hi', lower=-8.0, upper=-3.0, value=p0['log_tau_hi'])
#            a = pymc.Uniform('a', lower=0.9*np.array([-0.0018978657,-0.01669747315,-0.00507228575,-0.0058924686,-0.0008685198]), upper=1.1*np.array([0.0222362157,0.00528944015,0.00767281475,0.0052059286,0.0009839638]), size=decomp_poly+1)
#            a = pymc.MvNormal('a', mu=p0['a_mu']*np.ones(decomp_poly+1), tau=(1./(2*p0['a_sd'])**2)*np.eye(decomp_poly+1))        
#            a = pymc.MvNormal('a', mu=ccd_priors['a'], tau=(1./(1e-10)**2)*np.eye(decomp_poly+1))        

            a = pymc.Normal('a', mu=0, tau=1./(0.01**2), value=p0["a"], size=decomp_poly+1)
#            noise = pymc.Uniform('noise', lower=0., upper=1.)
            if self.guess_noise:
                noise_r = pymc.Uniform('noise_real', lower=0., upper=1.)
                noise_i = pymc.Uniform('noise_imag', lower=0., upper=1.)

#            noises = pymc.Lambda('noises', lambda noise=noise: np.reshape(noise, (2,1)))
            # Deterministics
    #        @pymc.deterministic(plot=False)
    #        def m_hi(mp_hi=mp_hi):
    #            return 10**mp_hi / (1 + 10**mp_hi)
            @pymc.deterministic(plot=False)
            def zmod(R0=R0, a=a):
                return Decomp_cyth(w, tau_10, log_taus, c_exp, R0, a)
            @pymc.deterministic(plot=False)
            def m_i(a=a):
                return np.sum((a*log_taus.T).T, axis=0)
#                return np.poly1d(a)(ccd_priors['log_tau'])
            
            
            
            @pymc.deterministic(plot=False)
            def total_m(m_i=m_i):
                return np.sum(m_i[(log_tau >= self.log_min_tau)&(m_i >= 0)&(log_tau <= max(log_tau)-1)])
#                return np.sum(m_i[(log_tau >= self.log_min_tau)&(m_i >= 0)&(log_tau <= 0)])
            @pymc.deterministic(plot=False)
            def log_half_tau(m_i=m_i):
                return log_tau[cond][np.where(np.cumsum(m_i[cond])/np.sum(m_i[cond]) > 0.5)[0][0]]
            @pymc.deterministic(plot=False)
            def log_peak_tau(m_i=m_i):
                cond = np.r_[True, m_i[1:] > m_i[:-1]] & np.r_[m_i[:-1] > m_i[1:], True]
                cond[0] = False
                try: return log_tau[cond][0]
                except: return log_tau[0]
            @pymc.deterministic(plot=False)
            def log_mean_tau(m_i=m_i):
                return np.log10(np.exp(old_div(np.sum(m_i[cond]*np.log(10**log_tau[cond])),np.sum(m_i[cond]))))
            # Likelihood
#            obs = pymc.Normal('obs', mu=zmod, tau=1./((self.data["zn_err"]+noise)**2), value=self.data["zn"], size = (2, len(w)), observed=True)
#            for i in range(2):
#                obs_i = pymc.Normal('obs_%s'%i, mu=zmod[i], tau=1./((self.data["zn_err"][i]+noise[i])**2), value=self.data["zn"][i], size = len(w), observed=True)
            if self.guess_noise:
                obs_r = pymc.Normal('obs_r', mu=zmod[0], tau=1./((noise_r)**2), value=self.data["zn"][0], size = len(w), observed=True)
                obs_i = pymc.Normal('obs_i', mu=zmod[1], tau=1./((noise_i)**2), value=self.data["zn"][1], size = len(w), observed=True)
            else:
                obs = pymc.Normal('obs', mu=zmod, tau=1./(self.data["zn_err"]**2), value=self.data["zn"], size = (2, len(w)), observed=True)

            return locals()
    
    #==============================================================================
        """
        Main section
        """
    #==============================================================================
        # Importing data
        self.data = get_data(self.filename, self.headers, self.ph_units)
        
        
        
        
        if (self.data["pha_err"] == 0).all():
            self.guess_noise = True
        seigle_m = (old_div((self.data["amp"][-1] - self.data["amp"][0]), self.data["amp"][-1]) ) # Estimating Seigel chargeability
        w = 2*np.pi*self.data["freq"] # Frequencies measured in rad/s
    #    n_freq = len(w)
    #    n_decades = np.ceil(max(np.log10(old_div(1.0,w)))) - np.floor(min(np.log10(old_div(1.0,w))))
        # Relaxation times associated with the measured frequencies (Debye decomposition only)
#        log_tau = self.ccd_priors['log_tau']
        if self.model == "PDecomp":
            log_tau = np.linspace(np.floor(min(np.log10(old_div(1.0,w)))-1), np.floor(max(np.log10(old_div(1.0,w)))+1), 50)
            cond = (log_tau >= min(log_tau)+1)&(log_tau <= max(log_tau)-1)
            log_taus = np.array([log_tau**i for i in list(reversed(range(0,self.decomp_poly+1)))]) # Polynomial approximation for the RTD
            tau_10 = 10**log_tau # Accelerates sampling
            self.data["tau"] = tau_10 # Put relaxation times in data dictionary
    
        # Time and date (for saving traces)
        sample_name = self.filename.replace("\\", "/").split("/")[-1].split(".")[0]
    #    actual_path = str(path.dirname(path.realpath(argv[0])))
        working_path = getcwd().replace("\\", "/")+"/"
        now = datetime.now()
        save_time = now.strftime('%Y%m%d_%H%M%S')
        save_date = now.strftime('%Y%m%d')
        out_path = '%s/Txt traces/%s/%s/%s-%s-%s/'%(working_path, save_date,
                                                     sample_name, self.model,
                                                     sample_name, save_time)
    
        """
        #==========================================================================
        Call to run_MCMC function
        #==========================================================================
        """
        # "ColeCole", "Dias", "Debye" or "Shin"
        sim_dict = {"ColeCole": {"func": ColeColeModel,     "args": [self.cc_modes]          },
                    "Dias":     {"func": DiasModel,         "args": []                  },
                    "PDecomp":  {"func": PolyDecompModel,   "args": [self.decomp_poly, self.c_exp, self.ccd_priors]},
                    "Shin":     {"func": ShinModel,         "args": []                  },
    #                "Custom":   {"func": YourModel,     "args": [opt_args]   },
                    "CCD":      {"func": stoCCD,            "args": [self.c_exp, self.ccd_priors]}
                    }
        simulation = sim_dict[self.model] # Pick entries for the selected model
        self.MDL = run_MCMC(simulation["func"](*simulation["args"]), self.mcmc, save_traces=self.keep_traces, save_where=out_path) # Run MCMC simulation with selected model and arguments
    #    if not keep_traces: rmtree(out_path)   # Deletes the traces if not wanted
    
        """
        #==========================================================================
        Results
        #==========================================================================
        """
        self.pm = format_results(self.MDL, self.data["Z_max"]) # Format output
        zmodstats = self.MDL.stats(chain=-1)["zmod"] # Take last chain
        zn_avg = zmodstats["mean"]
        zn_l95 = zmodstats["95% HPD interval"][0]
        zn_u95 = zmodstats["95% HPD interval"][1]
        avg = self.data["Z_max"]*(zn_avg[0] + 1j*zn_avg[1]) # (In complex notation, de-normalized)
        l95 = self.data["Z_max"]*(zn_l95[0] + 1j*zn_l95[1]) # (In complex notation, de-normalized)
        u95 = self.data["Z_max"]*(zn_u95[0] + 1j*zn_u95[1]) # (In complex notation, de-normalized)
        self.fit = {"best": avg, "lo95": l95, "up95": u95} # Best fit dict with 95% HDP
        self.model_type = {"log_min_tau":self.log_min_tau, "c_exp":self.c_exp, "decomp_polyn":self.decomp_poly, "cc_modes":self.cc_modes}
        # Output
#        return {"pymc_model": MDL, "params": pm, "data": data, "fit": fit, "SIP_model": model, "path": filename, "mcmc": mcmc, "model_type": {"log_min_tau":log_min_tau, "c_exp":c_exp, "decomp_polyn":decomp_poly, "cc_modes":cc_modes}}
        # End of inversion
                    
    
    #==============================================================================
"""
References:

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
#==============================================================================
