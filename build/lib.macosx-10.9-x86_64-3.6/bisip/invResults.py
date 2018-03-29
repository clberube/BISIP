# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 14:51:26 2015

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

This Python module contains functions to visualize the Bayesian inversion results
"""
from __future__ import division
from __future__ import print_function

#==============================================================================
from builtins import zip
from builtins import str
from builtins import range
from past.utils import old_div
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.pyplot import rcParams
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import LogLocator
from matplotlib.ticker import NullFormatter
import numpy as np
from os import path, makedirs
from os import getcwd
from math import ceil
from pymc import raftery_lewis, gelman_rubin, geweke
from scipy.stats import norm, gaussian_kde
from bisip.utils import get_data, get_model_type, save_figure
from bisip.utils import var_depth, flatten, find_nearest

import matplotlib as mpl
mpl.rc_file_defaults()

SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 14

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize

#==============================================================================
sym_labels = dict([('resi', r"$\rho$ ($\Omega$m)"),
                   ('freq', r"Frequency (Hz)"),
                   ('phas', r"-Phase (mrad)"),
                   ('ampl', r"$|\rho|$ (normalized)"),
                   ('real', r"$\rho$' (normalized)"),
                   ('imag', r"$-\rho$'' (normalized)"),
                   ('realrho', r"$\rho$' ($\Omega$m)"),
                   ('imagrho', r"$-\rho$'' ($\Omega$m)"),
                   ])

parlbl_dic = {'NRMSE_r': r"$\rho''_{\mathrm{NRMSE}}$",
              'NRMSE_i': r"$\rho'_{\mathrm{NRMSE}}$",
              'delta': r'$\delta$',
              'eta': r'$\eta$',
              'R0': r'$\rho_0/\rho_{max}$',
              'log_tau': r'$\log_{10}\tau$',
              'log_tau1': r'$\log_{10}\tau_1$',
              'log_tau2': r'$\log_{10}\tau_2$',
              'log_tau3': r'$\log_{10}\tau_3$',
              'c1': r'$c_1$',
              'c2': r'$c_2$',
              'c3': r'$c_3$',
              'm1': r'$m_1$',
              'm2': r'$m_2$',
              'm3': r'$m_3$',
              'a1': r'$a_0$',
              'a2': r'$a_1$',
              'a3': r'$a_2$',
              'a4': r'$a_3$',
              'a5': r'$a_4$',
              'a6': r'$a_5$',
              'a7': r'$a_6$',
              'total_m': r'$\Sigma m$',
              'log_total_m': r'$\log_{10}(\Sigma m)$',
              'log_mean_tau': r'$\log_{10}(\bar{\tau})$',
              'log_U_tau': r'$\log_{10}(U_{\tau})$',
              'log_half_tau': r'$\log_{10}(\tau_{50})$',
              'log_peak_tau': r'$\log_{10}(\tau_p)$',
              'log_peak_tau1': r'$\log_{10}(\tau_p1)$',
              'log_peak_tau2': r'$\log_{10}(\tau_p2)$',
              'log_peak_tau3': r'$\log_{10}(\tau_p3)$',
              'log_peak_tau4': r'$\log_{10}(\tau_p4)$',
              'log_peak_tau5': r'$\log_{10}(\tau_p5)$',
              'log_peak_m': r'$\log_{10}(m_p)$',
              'log_peak_m1': r'$\log_{10}(m_p1)$',
              'log_peak_m2': r'$\log_{10}(m_p2)$',
              'log_peak_m3': r'$\log_{10}(m_p3)$',
              'log_peak_m4': r'$\log_{10}(m_p4)$',
              'log_peak_m5': r'$\log_{10}(m_p5)$',
              'peak_m': r'$\log_{10}(m_p)$',
              'peak_m1': r'$\log_{10}(m_p1)$',
              'peak_m2': r'$\log_{10}(m_p2)$',
              'peak_m3': r'$\log_{10}(m_p3)$',
              'peak_m4': r'$\log_{10}(m_p4)$',
              'peak_m5': r'$\log_{10}(m_p5)$',
              'log_noise_m': r'$\log_{10}(\delta_{m})$',
              'log_noise_rho': r'$\log_{10}(\delta_{\rho})$',
              'log_noise_tau': r'$\log_{10}(\delta_{\tau})$',
              'm': r'$m$',
              'n1': r'$n_1$',
              'n2': r'$n_2$',
              'R1': r'$\rho_1$',
              'R2': r'$\rho_2$',
              'log_Q1': r'$\log_{10}(Q_1)$',
              'log_Q2': r'$\log_{10}(Q_2)$',
              }
    
default_ignore = [
                  'zmod', 'log_m_i', 'log_tau_i', 'cond', 'm_i', 
                  'peak_m', 'log_peak_tau', 'log_peak_m', 
                      ]

#==============================================================================
def print_resul(sol):
    """
    Prints the results to the console after inversion
    Pass mcmcinv object
    """
    pm, model, filename = sol.pm, sol.model, sol.filename
    print('\n\nInversion success!')
    print('Name of file:', filename)
    print('Model used:', model)
    try:
        pm.pop("cond_std")
        pm.pop("tau_i_std")
        pm.pop("m_i_std")
    except:
        pass
    e_keys = sorted([s for s in list(pm.keys()) if "_std" in s])
    v_keys = [e.replace("_std", "") for e in e_keys]
    labels = ["{:<8}".format(x+":") for x in v_keys]
    np.set_printoptions(formatter={'float': lambda x: format(x, '6.3E')})
    for l, v, e in zip(labels, v_keys, e_keys):
        if "noise" not in l:
            print(l, np.atleast_1d(pm[v]), '+/-', np.atleast_1d(pm[e]), np.char.mod('(%.2f%%)',abs(100*pm[e]/pm[v])))
        else:
            print(l, np.atleast_1d(pm[v]), '+/-', np.atleast_1d(pm[e]))
            
            
def plot_data(filename, headers, ph_units, save=False, 
              save_as_png=False, dpi=None, fig_nb=None):
    """
    Plots data before doing inversion
    Pass full file path, number of headers and phase units
    """
    ext = ['png' if save_as_png else 'pdf'][0]
    data = get_data(filename,headers,ph_units)

    # Graphiques du data
    Z = data["Z"]
    dZ = data["Z_err"]
    f = data["freq"]
    zn_dat = Z
    zn_err = dZ
    Pha_dat = 1000*data["pha"]
    Pha_err = 1000*data["pha_err"]
    Amp_dat = data["amp"]
    Amp_err = data["amp_err"]

    fig, ax = plt.subplots(2, 2, figsize=(8,5), sharex=True)
    # Real-Imag
    plt.axes(ax[0,0])
    plt.errorbar(f, zn_dat.real, zn_err.real, None, fmt='o', mfc='white', markersize=5, label='Data', zorder=0)
    ax[0,0].set_xscale("log")
    plt.ylabel(sym_labels['realrho'])
    
    plt.axes(ax[0,1])
    plt.errorbar(f, -zn_dat.imag, zn_err.imag, None, fmt='o', mfc='white', markersize=5, label='Data', zorder=0)
    ax[0,1].set_xscale("log")
    plt.ylabel(sym_labels['imagrho'])

    # Freq-Phas
    plt.axes(ax[1,1])
    plt.errorbar(f, -Pha_dat, Pha_err, None, fmt='o', mfc='white', markersize=5, label='Data', zorder=0)
    ax[1,1].set_yscale("log", nonposy='clip')
    ax[1,1].set_xscale("log")
    plt.xlabel(sym_labels['freq'])
    plt.ylabel(sym_labels['phas'])

    # Adjust for low or high phase response
    if  (-Pha_dat < 1).any() and (-Pha_dat >= 0.1).any():
        plt.ylim([0.1,10**np.ceil(max(np.log10(-Pha_dat)))])  
    if  (-Pha_dat < 0.1).any() and (-Pha_dat >= 0.01).any():
        plt.ylim([0.01,10**np.ceil(max(np.log10(-Pha_dat)))]) 
    
    # Freq-Ampl
    plt.axes(ax[1,0])
    plt.errorbar(f, Amp_dat, Amp_err, None, fmt='o', mfc='white', markersize=5, label='Data', zorder=0)
    ax[1,0].set_xscale("log")
    plt.xlabel(sym_labels['freq'])
    plt.ylabel(sym_labels['resi'])

    for a in ax.flat:
        a.grid('on')
        
    fig.tight_layout()

    if save: 
        fn = 'DAT-%s.%s'%(filename,ext)
        save_figure(fig, subfolder='Data', fname=fn, dpi=dpi)

    plt.close(fig)        
    return fig


def plot_fit(sol, save=False, draw=True, 
             save_as_png=False, dpi=None, fig_nb=""):
    """
    Plots the average fit and uncertainty
    Pass mcmcinv object (sol)
    """
    ext = ['png' if save_as_png else 'pdf'][0]

    # Prepare data for plotting
    f = sol.data["freq"]
    Zr0 = sol.data["Z_max"]
    zn_dat = sol.data["Z"]/Zr0
    zn_err = sol.data["Z_err"]/Zr0
    zn_fit = sol.fit["best"]/Zr0
    zn_min = sol.fit["lo95"]/Zr0
    zn_max = sol.fit["up95"]/Zr0
    
    Pha_dat = 1000*sol.data["pha"]
    Pha_err = 1000*sol.data["pha_err"]
    Pha_fit = 1000*np.angle(sol.fit["best"])
    Pha_min = 1000*np.angle(sol.fit["lo95"])
    Pha_max = 1000*np.angle(sol.fit["up95"])
    
    Amp_dat = sol.data["amp"]/Zr0
    Amp_err = sol.data["amp_err"]/Zr0
    Amp_fit = abs(sol.fit["best"])/Zr0
    Amp_min = abs(sol.fit["lo95"])/Zr0
    Amp_max = abs(sol.fit["up95"])/Zr0
    
    fig, ax = plt.subplots(2, 2, figsize=(8,5), sharex=True)
    
    # Freq-Imag
    plt.sca(ax[0,0])
    plt.errorbar(f, -zn_dat.imag, zn_err.imag, None, color='k', fmt='o', mfc='white', markersize=5, label='Data', zorder=0)
    p=plt.plot(f, -zn_fit.imag, ls='-', label="Model",zorder=2)
    plt.fill_between(f, -zn_max.imag, -zn_min.imag, alpha=0.4, color=p[0].get_color(), zorder=1, label='95% HPD')
    plt.ylabel(sym_labels['imag'])
    plt.legend(loc='best', labelspacing=0.2, handlelength=1, framealpha=1)
    
    # Freq-Real
    plt.sca(ax[0,1])
    plt.errorbar(f, zn_dat.real, zn_err.real, None, color='k', fmt='o', mfc='white', markersize=5, label='Data', zorder=0)
    p=plt.plot(f, zn_fit.real, ls='-', label="Model",zorder=2)
    plt.fill_between(f, zn_max.real, zn_min.real, alpha=0.4, color=p[0].get_color(), zorder=1, label='95% HPD')
    plt.ylabel(sym_labels['imag'])
    plt.legend(loc='best', labelspacing=0.2, handlelength=1, framealpha=1)
    
    # Freq-Phas
    plt.sca(ax[1,0])
    plt.errorbar(f, -Pha_dat, Pha_err, None, fmt='o', color='k', mfc='white', markersize=5, label='Data', zorder=0)
    p=plt.plot(f, -Pha_fit, ls='-', label='Model', zorder=2)
    ax[1,0].set_yscale("log", nonposy='clip')
    plt.xscale('log')
    plt.fill_between(f, -Pha_max, -Pha_min, color=p[0].get_color(), alpha=0.4, zorder=1, label='95% HPD')
    plt.xlabel(sym_labels['freq'])
    plt.ylabel(sym_labels['phas'])
    plt.legend(loc='best', labelspacing=0.2, handlelength=1, framealpha=1)

    # Freq-Ampl
    plt.sca(ax[1,1])
    plt.errorbar(f, Amp_dat, Amp_err, None, fmt='o', color='k', mfc='white', markersize=5, label='Data', zorder=0)
    p=plt.semilogx(f, Amp_fit, ls='-', label='Model', zorder=2)
    plt.fill_between(f, Amp_max, Amp_min, color=p[0].get_color(), alpha=0.4, zorder=1, label='95% HPD')
    plt.xscale('log')
    plt.xlabel(sym_labels['freq'])
    plt.ylabel(sym_labels['ampl'])
    plt.legend(loc='best', labelspacing=0.2, handlelength=1, framealpha=1)

    for a in ax.flat:
        a.grid(True)

    plt.tight_layout(pad=0, h_pad=0.5, w_pad=1)
        
    if save:
        fn = '%sFIT-%s-%s.%s'%(fig_nb,sol.model_type_str,sol.filename,ext)
        save_figure(fig, subfolder='Fit figures', fname=fn, dpi=dpi)

    plt.close(fig)        
    if draw:    return fig
    else:       return None
            
def plot_histo(sol, save=False, draw=True, save_as_png=False, dpi=None, 
               ignore=default_ignore,
               ):    
    """
    Plots the traces of stochastic and
    deterministic parameters in mcmcinv object (sol)
    Ignores the ones in list argument ignore
    """
    # Get some settings
    ext = ['png' if save_as_png else 'pdf'][0] # get figure format  
    
    # Get all variable names from mcmcinv object
    headers = sorted(sol.trace_dict.keys())
    # Remove unwanted headers
    headers = [h for h in headers if h.strip('0123456789') not in ignore]
    # Extract the needed traces
    traces = [sol.trace_dict[h] for h in headers]
    
    # Subplot settings
    ncols = 2
    nrows = int(ceil(len(headers)*1.0 / ncols))
    fig, ax = plt.subplots(nrows, ncols, figsize=(8,nrows*1.8))

    # Plot histograms
    for i in range(len(headers)):
        data = sorted(traces[i])
        plt.sca(ax.flat[i])
        plt.xlabel(parlbl_dic[headers[i]])
        try:
            hist = plt.hist(data, bins=20, histtype='stepfilled', density=False, linewidth=1.0, color='0.95', alpha=1)
            plt.hist(data, bins=20, histtype='step', density=False, linewidth=1.0, alpha=1)
            fit = norm.pdf(data, np.mean(data), np.std(data))                
            xh = [0.5 * (hist[1][r] + hist[1][r+1]) for r in range(len(hist[1])-1)]
            binwidth = (max(xh) - min(xh)) / len(hist[1])
            fit *= len(data) * binwidth
            plt.plot(data, fit, "-", color='k', linewidth=1)
        except:
            print("File %s: failed to plot %s histogram.\nNot enough accepted moves." %(sol.filename,headers[i]))
        plt.grid(False)
        plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
        
    for c in range(nrows):
        ax[c][0].set_ylabel("Frequency")
    for a in ax.flat[ax.size - 1:len(headers) - 1:-1]:
        a.set_visible(False)
    plt.tight_layout(pad=1, w_pad=1, h_pad=0)
        
    if save: 
        fn = 'HST-%s-%s.%s'%(sol.model_type_str,sol.filename,ext)
        save_figure(fig, subfolder='Histograms', fname=fn, dpi=dpi)
    
    plt.close(fig)        
    if draw:    return fig
    else:       return None

def plot_traces(sol, save=False, draw=True, save_as_png=False, dpi=None, 
                ignore=default_ignore,
                ):
    """
    Plots the traces of stochastic and
    deterministic parameters in mcmcinv object (sol)
    Ignores the ones in list argument ignore
    """
    # Get some settings
    ext = ['png' if save_as_png else 'pdf'][0] # get figure format  
    mcmc = sol.mcmc # get MCMC parameters
    
    # Get all variable names from mcmcinv object
    headers = sorted(sol.trace_dict.keys()) # In alphabetical order
    # Remove unwanted headers
    headers = [h for h in headers if h.strip('0123456789') not in ignore]
    # Extract the needed traces
    traces = [sol.trace_dict[h] for h in headers]
    
    # Subplot settings
    ncols = 2
    nrows = int(ceil(len(headers)*1.0 / ncols))
    fig, ax = plt.subplots(nrows, ncols, figsize=(8, nrows*1.5), sharex=True)

    # Plot traces
    for i in range(len(headers)):
        data = traces[i]
        x = np.arange(mcmc["nb_burn"]+1, mcmc["nb_iter"]+1, mcmc["thin"])
        plt.sca(ax.flat[i])
        plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
        plt.ylabel(parlbl_dic[headers[i]])    
        plt.plot(x, data,'-', alpha=0.8)
        plt.plot(x, np.mean(data)*np.ones(len(x)), color='k',linestyle='--', linewidth=2)
        if mcmc["nb_burn"] == 0:
            plt.xscale('log')
        else:
            plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))    
        plt.grid(False)
    plt.tight_layout(pad=0, w_pad=0.5, h_pad=0)
    
    for a in ax[-1]:
        a.set_xlabel("Iteration number")
        
    if save: 
        fn = 'TRA-%s-%s.%s'%(sol.model_type_str,sol.filename,ext)
        save_figure(fig, subfolder='Traces', fname=fn, dpi=dpi)

    plt.close(fig)        
    if draw:    return fig
    else:       return None

def plot_KDE(sol, var1, var2, fig=None, ax=None, draw=True, save=False, save_as_png=False, dpi=None):
    """
    Like the hexbin plot but a 2D KDE
    Pass mcmcinv object and 2 variable names as strings
    """
    ext = ['png' if save_as_png else 'pdf'][0]
    if fig == None or ax == None:
        fig, ax = plt.subplots(figsize=(3,3))
    MDL = sol.MDL
    if var1 == "R0":
        stoc1 = "R0"
    else:
        stoc1 =  ''.join([i for i in var1 if not i.isdigit()])
        stoc_num1 = [int(i) for i in var1 if i.isdigit()]
    try:
        x = MDL.trace(stoc1)[:,stoc_num1[0]-1]
    except:
        x = MDL.trace(stoc1)[:]
    if var2 == "R0":
        stoc2 = "R0"
    else:
        stoc2 =  ''.join([i for i in var2 if not i.isdigit()])
        stoc_num2 = [int(i) for i in var2 if i.isdigit()]
    try:
        y = MDL.trace(stoc2)[:,stoc_num2[0]-1]
    except:
        y = MDL.trace(stoc2)[:]
    xmin, xmax = min(x), max(x)
    ymin, ymax = min(y), max(y) 
    # Peform the kernel density estimate
    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([x, y])
    kernel = gaussian_kde(values)
    kernel.set_bandwidth(bw_method='silverman')
#        kernel.set_bandwidth(bw_method=kernel.factor * 2.)
    f = np.reshape(kernel(positions).T, xx.shape)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    plt.sca(ax)
    # Contourf plot
    plt.grid(None)
    plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
    plt.xticks(rotation=90)
    plt.locator_params(axis = 'y', nbins = 7)
    plt.locator_params(axis = 'x', nbins = 7)
    ax.contourf(xx, yy, f, cmap=plt.cm.viridis, alpha=0.8)
    ax.scatter(x, y, color='k', s=1, zorder=2)

    plt.ylabel("%s" %var2)
    plt.xlabel("%s" %var1)

    if save: 
        fn = 'KDE-%s-%s.%s'%(sol.model_type_str,sol.filename,ext)
        save_figure(fig, subfolder='2D-KDE', fname=fn, dpi=dpi)
    
    plt.close(fig)        
    if draw:    return fig
    else:       return None

def plot_hexbin(sol, var1, var2, draw=True, save=False, save_as_png=False, dpi=None):
    """
    Like the 2D KDE plot but a hexbin
    Pass mcmcinv object and 2 variable names as strings
    """
    ext = ['png' if save_as_png else 'pdf'][0]
    MDL = sol.MDL
    if var1 == "R0":
        stoc1 = "R0"
    else:
        stoc1 =  ''.join([i for i in var1 if not i.isdigit()])
        stoc_num1 = [int(i) for i in var1 if i.isdigit()]
    try:
        x = MDL.trace(stoc1)[:,stoc_num1[0]-1]
    except:
        x = MDL.trace(stoc1)[:]
    if var2 == "R0":
        stoc2 = "R0"
    else:
        stoc2 =  ''.join([i for i in var2 if not i.isdigit()])
        stoc_num2 = [int(i) for i in var2 if i.isdigit()]
    try:
        y = MDL.trace(stoc2)[:,stoc_num2[0]-1]
    except:
        y = MDL.trace(stoc2)[:]
    xmin, xmax = min(x), max(x)
    ymin, ymax = min(y), max(y)
    fig, ax = plt.subplots(figsize=(4,3))
    plt.grid(None)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    plt.hexbin(x, y, gridsize=15, cmap=plt.cm.magma_r)
    plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
    plt.xticks(rotation=90)
    plt.locator_params(axis = 'y', nbins = 5)
    plt.locator_params(axis = 'x', nbins = 5)    
    cb = plt.colorbar()
    cb.set_label('Number of observations')
    plt.ylabel("%s" %var2)
    plt.xlabel("%s" %var1)

    if save: 
        fn = 'HEX-%s-%s.%s'%(sol.model_type_str,sol.filename,ext)
        save_figure(fig, subfolder='Hexbins', fname=fn, dpi=dpi)

    plt.close(fig)        
    if draw:    return fig
    else:       return None


def plot_summary(sol, save=False, draw=True, save_as_png=False, dpi=None,
                 ignore=default_ignore,
                 fig_nb="",
                 ):
    """
    Plots a parameter summary and 
    Gelman-Rubin R-hat for multiple chains
    """
    
    ext = ['png' if save_as_png else 'pdf'][0]
    ch_nb = sol.mcmc["nb_chain"]

    keys = sorted([k for k in sol.var_dict.keys() if k not in ignore])        
    trac = [[sol.var_dict[x].trace(chain=n).mean(axis=0) for x in keys] for n in range(ch_nb)]
    deps = [var_depth(sol.var_dict[x]) for x in keys]
    lbls = list(reversed(flatten([[k+'%s'%(x+1) for x in range(d)] if d > 1 else k for k, d in zip(keys,deps)])))
    
    if ch_nb >= 2:
        rhat = [gelman_rubin([sol.MDL.trace(var, -x)[:] for x in range(sol.mcmc['nb_chain'])]) for var in keys]
        R = np.array(flatten(rhat))
        R[R > 5] = 5 
    else:
        print("\nTwo or more chains of equal length required for Gelman-Rubin convergence")
        R = len(lbls)*[None]
        
    fig, axes = plt.subplots(figsize=(6,4))
    gs2 = gridspec.GridSpec(3, 3)
    ax1 = plt.subplot(gs2[:, :-1])
    ax2 = plt.subplot(gs2[:, -1], sharey = ax1)
    for i in range(len(lbls)):
        for c in range(ch_nb):
            val_m = np.array(flatten(trac[c]))
            ax1.scatter(val_m[i], len(val_m)-(i+1) , color="C0", marker=".", 
                        s=50, facecolor='k', edgecolors='k',alpha=1)
        ax2.scatter(R[i], i, color="C3", marker="<", s=50, alpha=1)

    ax1.set_ylim([-1, len(lbls)])
    ax1.set_yticks(list(range(0,len(lbls))))
    ax1.set_yticklabels([parlbl_dic[l] for l in lbls])
    ax1.set_axisbelow(True)
    ax1.yaxis.grid(True)
    ax1.xaxis.grid(False)
    ax1.set_xlim(ax1.get_xlim())
    ax1.set_xlabel(r'Parameter value')

    plt.setp(ax2.get_yticklabels(), visible=False)
    ax2.set_xlim([0.5, 5.5])
    ax2.set_xticklabels(["","1","2","3","4","5+"])
    ax2.set_xticks([0.5, 1, 2, 3, 4, 5, ])
    ax2.set_axisbelow(True)
    ax2.yaxis.grid(True)
    ax2.xaxis.grid(False)
    ax2.set_xlabel(r'$\hat{R}$')
    ax2.axvline(1, ls='--', color='C0', zorder=0)

    plt.tight_layout()
    plt.close(fig)        

    if save: 
        fn = '%sSUM-%s-%s.%s'%(fig_nb,sol.model_type_str,sol.filename,ext)
        save_figure(fig, subfolder='Summaries', fname=fn, dpi=dpi)

    if draw:    return fig
    else:       return None

def plot_autocorr(sol, save=False, draw=True, save_as_png=False, dpi=None,
                 ignore=default_ignore,
                 ):
    """
    Plots autocorrelations
    """
    ext = ['png' if save_as_png else 'pdf'][0]
    MDL = sol.MDL
    
    keys = [k for k in sol.var_dict.keys() if k not in ignore]

    for (i, k) in enumerate(keys):
        vect = old_div((MDL.trace(k)[:].size),(len(MDL.trace(k)[:])))
        if vect > 1:
         keys[i] = [k+"%d"%n for n in range(1,vect+1)]
    keys = list(flatten(keys))
    ncols = 2
    nrows = int(ceil(len(keys)*1.0 / ncols))
    fig, ax = plt.subplots(nrows, ncols, figsize=(10,nrows*2))
    plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
    for (a, k) in zip(ax.flat, keys):
        if k[-1] not in ["%d"%d for d in range(1,8)] or k =="R0":
            data = sorted(MDL.trace(k)[:].ravel())
        else:
            data = sorted(MDL.trace(k[:-1])[:][:,int(k[-1])-1].ravel())
        plt.sca(a)
        plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)
        plt.gca().get_xaxis().get_major_formatter().set_useOffset(False)
        plt.yticks(fontsize=12)
        plt.xticks(fontsize=12)
        plt.ylabel(k, fontsize=12)
        to_thin = old_div(len(data),50)
        if to_thin != 0: plt.xlabel("Lags / %d"%to_thin, fontsize=12)
        else: plt.xlabel("Lags", fontsize=12)
        max_lags = None
        if len(data) > 50: data= data[::to_thin]
        plt.acorr(data, usevlines=True, maxlags=max_lags, detrend=plt.mlab.detrend_mean)
        plt.grid(None)
    fig.tight_layout()
    for a in ax.flat[ax.size - 1:len(keys) - 1:-1]:
        a.set_visible(False)
        
    if save: 
        fn = 'AC-%s-%s.%s'%(sol.model_type_str,sol.filename,ext)
        save_figure(fig, subfolder='Autocorrelations', fname=fn, dpi=dpi)

    plt.close(fig)        
    if draw:    return fig
    else:       return None

def plot_rtd(sol, save=False, draw=True, save_as_png=False, dpi=None):
    """
    Plots the relaxation time distribution (RTD)
    for a polynomial decomposition or ccdt results
    """
    ext = ['png' if save_as_png else 'pdf'][0]
    fig, ax = plt.subplots(figsize=(4,3))
    try:
        bot95 = 10**sol.MDL.stats()["log_m_i"]['95% HPD interval'][0]
        top95 = 10**sol.MDL.stats()["log_m_i"]['95% HPD interval'][1]
        log_tau = 10**sol.MDL.stats()["log_tau_i"]['mean']
        log_m = 10**sol.MDL.stats()["log_m_i"]['mean']
    except:
        bot95 = sol.MDL.stats()["m_i"]['95% HPD interval'][0]
        top95 = sol.MDL.stats()["m_i"]['95% HPD interval'][1]
        log_tau = 10**sol.MDL.log_tau
        log_m = sol.MDL.stats()["m_i"]['mean']            
    plt.errorbar(log_tau, log_m, None, None, color="C7", linestyle='-', label="RTD")
    try:
        peaks = 10**np.atleast_1d(sol.MDL.stats()["log_peak_tau"]["mean"])
        uncer_peaks = 10**sol.MDL.stats()["log_peak_tau"]['95% HPD interval'].T.reshape(len(np.atleast_1d(sol.MDL.stats()["log_peak_tau"]['mean'])),2)
        m_peaks = log_m[[list(log_tau).index(find_nearest(log_tau, peaks[x])) for x in range(len(peaks))]]
        if len(peaks) >= 1:
            plt.errorbar(peaks, m_peaks*1.2, None, None, color="C3", marker="v", markersize=5, linestyle="", label=r"$\tau_{peak}$")
            for i, u in enumerate(uncer_peaks):
                plt.axvspan(u[0], u[1], alpha=0.2, color="C3")
    except:
        pass
    plt.axvline(10**sol.MDL.stats()["log_half_tau"]['mean'],color="C0",linestyle=':', label=r"$\tau_{50}$")
    plt.axvline(10**sol.MDL.stats()["log_mean_tau"]['mean'],color='C2',linestyle='--', label=r"$\bar{\tau}$")
    inter = 10**sol.MDL.stats()["log_half_tau"]['95% HPD interval']
    plt.axvspan(inter[0], inter[1], alpha=0.2, color="C0")
    inter = 10**sol.MDL.stats()["log_mean_tau"]['95% HPD interval']
    plt.axvspan(inter[0], inter[1], alpha=0.2, color='C2')
    plt.axvspan(min(log_tau), min(log_tau)*10, alpha=0.1, color='C7')
    plt.axvspan(max(log_tau)/10, max(log_tau), alpha=0.1, color='C7')
    plt.fill_between(log_tau, bot95, top95, color="C7", alpha=0.2)
    plt.xlim([10**np.ceil(np.log10(min(log_tau))), 10**np.floor(np.log10(max(log_tau)))])
    ax.set_xlabel(r'$\tau$ (s)')
    ax.set_ylabel(r'$m$')
    plt.grid(False)
    plt.legend(fontsize=9, loc=1,labelspacing=0.2, handlelength=1.5)
    plt.xscale('log')
    plt.yscale('log', nonposy='clip')
    fig.tight_layout()
    if save: 
        fn = 'RTD-%s-%s.%s'%(sol.model_type_str,sol.filename,ext)
        save_figure(fig, subfolder='RTD', fname=fn, dpi=dpi)

    plt.close(fig)        
    if draw:    return fig
    else:       return None

def plot_deviance(sol, save=False, draw=True, save_as_png=False, dpi=None):
    """
    Plots the model deviance trace
    """
    ext = ['png' if save_as_png else 'pdf'][0]
    fig, ax = plt.subplots(figsize=(4,3))
    deviance = sol.MDL.trace('deviance')[:]
    sampler_state = sol.MDL.get_state()["sampler"]
    x = np.arange(sampler_state["_burn"]+1, sampler_state["_iter"]+1, sampler_state["_thin"])
    plt.plot(x, deviance, "-", color="C3", label="DIC = %d\nBPIC = %d" %(sol.MDL.DIC,sol.MDL.BPIC))
    plt.xlabel("Iteration")
    plt.ylabel("Model deviance")
    plt.legend(numpoints=1, loc="best", fontsize=9)
    plt.grid('on')
    if sampler_state["_burn"] == 0:
        plt.xscale('log')
    else:
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()
    
    if save: 
        fn = 'MDEV-%s-%s.%s'%(sol.model_type_str,sol.filename,ext)
        save_figure(fig, subfolder='ModelDeviance', fname=fn, dpi=dpi)

    plt.close(fig)        
    if draw:    return fig
    else:       return None

def logp_trace(model):
    """
    return a trace of logp for model
    """
    #init
    db = model.db
    n_samples = db.trace('deviance').length()
    logp = np.empty(n_samples, np.double)
    #loop over all samples
    for i_sample in range(n_samples):
        #set the value of all stochastic to their 'i_sample' value
        for stochastic in model.stochastics:
            try:
                value = db.trace(stochastic.__name__)[i_sample]
                stochastic.value = value

            except KeyError:
                print("No trace available for %s. " % stochastic.__name__)

        #get logp
        logp[i_sample] = model.logp
    return logp

def plot_logp(sol, save=False, draw=True, save_as_png=False, dpi=None):
    """
    Plots the model log-likelihood
    """
    ext = ['png' if save_as_png else 'pdf'][0]
    fig, ax = plt.subplots(figsize=(4,3))
    logp = logp_trace(sol.MDL)
    sampler_state = sol.MDL.get_state()["sampler"]
    x = np.arange(sampler_state["_burn"]+1, sampler_state["_iter"]+1, sampler_state["_thin"])
    plt.plot(x, logp, "-")
    plt.xlabel("Iteration")
    plt.ylabel("Log-likelihood")
    plt.grid('on')
    if sampler_state["_burn"] == 0:
        plt.xscale('log')
    else:
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()
    
    if save: 
        fn = 'LOGP-%s-%s.%s'%(sol.model_type_str,sol.filename,ext)
        save_figure(fig, subfolder='LogLikelihood', fname=fn, dpi=dpi)

    plt.close(fig)        
    if draw:    return fig
    else:       return None

def save_csv_traces(sol):
    """
    Saves the traces contained in mcmcinv 
    object sol to a single csv file.
    call with sol.save_csv_traces()
    """
    
    # Decide where to save the csv
    save_where = '/TraceResults/'
    working_path = getcwd().replace("\\", "/")+"/"
    save_path = working_path + save_where + "%s/"%sol.filename # Add subfolder for the sample
    
    # Get stochastics and deterministics
#    sto = MDL.stochastics
#    det = MDL.deterministics
    # Get names of parameters for save file
    pm_names = sorted(sol.var_dict.keys())
    
    # Get all stochastic and deterministic variables
    trl = [sol.var_dict[x] for x in pm_names]

    # Concatenate all traces in 1 matrix
    trace_mat = np.hstack([t.trace().reshape(-1, var_depth(t)) for t in trl])
    # Insert normalization resistivity in first column
    trace_mat = np.insert(trace_mat, 0, sol.data['Z_max'], axis=1) 

    # Get numbers for each subheader
    num_names = [var_depth(v) for v in trl]    
    # Make list of headers
    headers = [['%s%d'%(pm_names[p],x+1) for x in range(num_names[p])] if num_names[p] > 1 else [pm_names[p]] for p in range(len(pm_names))]
    
    # Change zmod numbers for real and imaginary parts
    for h, head in enumerate(headers):
        if 'zmod' in head[0]:   
            headers[h] = [head[i]+'.real' for i in range(int(len(head)/2))] + [head[i]+'.imag' for i in range(int(len(head)/2))]    
    
    flat_headers = [item for sublist in headers for item in sublist] # Flaten list 
    flat_headers.insert(0, 'rho_max') # Insert normalization resistivity in first column
    header = ','.join(flat_headers) # Join list into csv string 
    
    # Do the saving
    print("\nSaving CSV traces in:\n", save_path)
    if not path.exists(save_path):
        makedirs(save_path)
    np.savetxt(save_path+'TRACES_%s-%s_%s.csv' %(sol.model,sol.model_type_str,sol.filename), trace_mat, delimiter=',', header=header, comments="")

def save_resul(sol):
    
    # To do: rewrite with new mcmcinv object attributes
    # Fonction pour enregistrer les résultats
    MDL, pm = sol.MDL, sol.pm
    model = sol.model_type_str
    sample_name = sol.filename
    save_where = '/Results/'
    working_path = getcwd().replace("\\", "/")+"/"
    save_path = working_path+save_where+"%s/"%sample_name
    print("\nSaving csv file in:\n", save_path)
    if not path.exists(save_path):
        makedirs(save_path)
    if sol.model == 'PDecomp': 
        tag = 0
    else: 
        tag = 1
    A = []
    B = []
    headers = []
    keys = sorted(pm.keys())
    if sol.model in ["CCD", "PDecomp"]:
        keys += [keys.pop(keys.index("peak_tau"))] # Move to end
        keys += [keys.pop(keys.index("peak_m"))] # Move to end

    keys = [k for k in keys if "_std" not in k]
        
    for c, key in enumerate(keys):
        
        A.append(list(np.array(pm[key]).ravel()))
        B.append(list(np.array(pm[key+"_std"]).ravel()))        

        length = len(np.atleast_1d(pm[key]))

        if length > 1:
            for i in range(len(A[c])):
                headers.append(model+"_"+key+"_%d" %(i+tag))
                headers.append(model+"_"+key+("_%d"%(i+tag))+"_std")
        else:           
            if (key == "peak_tau")|(key == "peak_m"):
                headers.append(model+"_"+key+"_1")
                headers.append(model+"_"+key+"_1"+"_std")                
            else: 
                headers.append(model+"_"+key)
                headers.append(model+"_"+key+"_std")

    A=flatten(A)
    B=flatten(B)

    results = [None]*(len(A)+len(B))
    results[::2] = A
    results[1::2] = B

    headers = ','.join(headers)
    results = np.array(results)

    if sol.model == 'PDecomp': 
        tau_ = sol.data["tau"]
        add = ["%s_tau"%model+"%d"%(i) for i in range(len(tau_))]
        add = ','.join(add) + ',' 
        headers = add+headers
        results = np.concatenate((tau_,results))
    headers = "Z_max,Input_c_exponent," + headers
    results = np.concatenate((np.array([sol.data["Z_max"]]),np.array([sol.c_exp]),results))
    np.savetxt(save_path+'INV_%s-%s_%s.csv' %(sol.model,model,sample_name), results[None],
               header=headers, comments='', delimiter=',')
    vars_ = ["%s"%x for x in MDL.stochastics]+["%s"%x for x in MDL.deterministics]
    if "zmod" in vars_: vars_.remove("zmod")
    MDL.write_csv(save_path+'STATS_%s-%s_%s.csv' %(sol.model,model,sample_name), variables=(vars_))

def merge_results(sol,files):
    """
    Merge a batch of csv files to a single one
    """
    import pandas as pd
    save_where = '/Batch results/'
    working_path = getcwd().replace("\\", "/")+"/"
    save_path = working_path+save_where
    print("\nMerging csv files")
    if not path.exists(save_path):
        makedirs(save_path)
    dfs = [pd.read_csv(working_path+"/Results/%s/INV_%s-%s_%s.csv" %(f,sol.model,sol.model_type_str,f)) for f in files]
    listed_dfs = [list(d) for d in dfs]
    df_tot = pd.concat(dfs, axis=0)
    longest = max(enumerate(listed_dfs), key = lambda tup: len(tup[1]))[0]
    df_tot['Sample_ID'] = files
    df_tot.set_index('Sample_ID', inplace=True)
    df_tot = df_tot[dfs[longest].columns]
    df_tot.to_csv(save_path+"Merged_%s-%s_%s_TO_%s.csv" %(sol.model,sol.model_type_str,files[0],files[-1]))
    print("Batch file successfully saved in:\n", save_path)


def print_diagn(M, q, r, s):
    return raftery_lewis(M, q, r, s, verbose=0)

def plot_par():
    rc = {
          u'figure.edgecolor': 'white',
          u'figure.facecolor': 'white',
          u'savefig.bbox': u'tight',
          u'savefig.directory': u'~',
          
          u'savefig.edgecolor': u'white',
          u'savefig.facecolor': u'white',
          u'axes.formatter.use_mathtext': True,
          u'xtick.direction'  : 'in',
          u'ytick.direction'  : 'in',
          
          }
    return rc
rcParams.update(plot_par())