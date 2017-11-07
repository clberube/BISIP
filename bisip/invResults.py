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
from past.builtins import basestring
from past.utils import old_div
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.pyplot import rcParams
from matplotlib.ticker import MaxNLocator
import numpy as np
from os import path, makedirs
from os import getcwd
from math import ceil
from pymc import raftery_lewis, gelman_rubin, geweke
from scipy.stats import norm, gaussian_kde
from bisip.utils import get_data


#==============================================================================
sym_labels = dict([('resi', r"$\rho\/(\Omega\cdot m)$"),
                   ('freq', r"Frequency (Hz)"),
                   ('phas', r"-Phase (mrad)"),
                   ('ampl', r"$|\rho|$ (normalized)"),
                   ('real', r"$\rho$' (normalized)"),
                   ('imag', r"$-\rho$'' (normalized)")])

def flatten(x):
    result = []
    for el in x:
        if hasattr(el, "__iter__") and not isinstance(el, basestring):
            result.extend(flatten(el))
        else:
            result.append(el)
    return result

def get_model_type(sol):
    model = sol.model
    if model in ["PDecomp", "CCD"]:
        if sol.model_type["c_exp"] == 0.5:
            model = "WarburgDecomp"
        elif sol.model_type["c_exp"] == 1.0:
            model = "DebyeDecomp"
        else:
            model = "ColeColeDecomp"
        
        model = ''.join([c for c in model if c.isupper()])
        
    return model

def print_resul(sol):
#==============================================================================
    # Impression des résultats
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
            
def plot_histo(sol, no_subplots=False, save=False, save_as_png=True, fig_dpi=144):
    if save_as_png:
        save_as = 'png'
    else:
        save_as = 'pdf'
    MDL = sol.MDL
    model = sol.get_model_type()
    filename = sol.filename.replace("\\", "/").split("/")[-1].split(".")[0]
    keys = sorted([x.__name__ for x in MDL.deterministics]) + sorted([x.__name__ for x in MDL.stochastics])
    try:
        keys.remove("zmod")
        keys.remove("log_m_i")
        keys.remove("log_tau_i")
        keys.remove("cond")
#        keys.remove("log_half_tau")
#        keys.remove("log_peak_tau")
    except:
        pass
    for (i, k) in enumerate(keys):
        vect = old_div((MDL.trace(k)[:].size),(len(MDL.trace(k)[:])))
        if vect > 1:
            if "Decomp" in model:
                keys[i] = [k+"%d"%n for n in range(0,vect)]
            else:
                keys[i] = [k+"%d"%n for n in range(1,vect+1)]
    keys = list(flatten(keys))
    if no_subplots:
        figs = {}
        for c, k in enumerate(keys):
            fig, ax = plt.subplots(figsize=(6,4))
            if k == "R0":
                stoc = "R0"
            else:
                stoc =  ''.join([i for i in k if not i.isdigit()])
                stoc_num = [int(i) for i in k if i.isdigit()]
            try:
                data = sorted(MDL.trace(stoc)[:][:,stoc_num[0]-1])
            except:
                data = sorted(MDL.trace(stoc)[:])
            plt.xlabel("%s value"%k)
            plt.ylabel("Probability density")
            hist = plt.hist(data, bins=20, normed=True, linewidth=1.0, color="white")
            fit = norm.pdf(data, np.mean(data), np.std(data))
            plt.plot(data, fit, "-", label="Fitted PDF", linewidth=1.5)
            plt.legend(loc='best')
            plt.grid('off')
            if save:
                save_where = '/Figures/Histograms/%s/' %filename
                working_path = getcwd().replace("\\", "/")+"/"
                save_path = working_path+save_where
                if c == 0:
                    print("\nSaving histogram figures in:\n", save_path)
                if not path.exists(save_path):
                    makedirs(save_path)
                fig.savefig(save_path+'Histo-%s-%s-%s.%s'%(model,filename,k,save_as), bbox_inches='tight')
            figs[k] = fig
            plt.close(fig)
        return figs

    else:
        ncols = 2
        nrows = int(ceil(len(keys)*1.0 / ncols))
        fig, ax = plt.subplots(nrows, ncols, figsize=(7,nrows*1.8))
        for c, (a, k) in enumerate(zip(ax.flat, keys)):
            if k == "R0":
                stoc = "R0"
            else:
                stoc =  ''.join([i for i in k if not i.isdigit()])
                stoc_num = [int(i) for i in k if i.isdigit()]
            try:
                data = sorted(MDL.trace(stoc)[:][:,stoc_num[0]-1])
            except:
                data = sorted(MDL.trace(stoc)[:])
            plt.axes(a)
            plt.locator_params(axis = 'y', nbins = 7)
            plt.locator_params(axis = 'x', nbins = 6)
            plt.xlabel(k)
            try:
                hist = plt.hist(data, bins=20, normed=False, label=filename, edgecolor='#1f77b4', linewidth=1.0, color='#1f77b4', alpha=0.3)
                fit = norm.pdf(data, np.mean(data), np.std(data))                
                xh = [0.5 * (hist[1][r] + hist[1][r+1]) for r in range(len(hist[1])-1)]
                binwidth = old_div((max(xh) - min(xh)), len(hist[1]))
                fit *= len(data) * binwidth
                plt.plot(data, fit, "-", color='#ff7f0e', linewidth=1.5)
            except:
                print("File %s: could not plot %s histogram. Parameter is unstable (see trace)." %(filename,k))
            plt.grid('off')
            plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
        
        for c in range(nrows):
            ax[c][0].set_ylabel("Frequency")

        plt.tight_layout(pad=1, w_pad=1, h_pad=0)
        for a in ax.flat[ax.size - 1:len(keys) - 1:-1]:
            a.set_visible(False)
        if save:
            save_where = '/Figures/Histograms/'
            working_path = getcwd().replace("\\", "/")+"/"
            save_path = working_path+save_where
            print("\nSaving parameter histograms in:\n", save_path)
            if not path.exists(save_path):
                makedirs(save_path)
            fig.savefig(save_path+'Histo-%s-%s.%s'%(model,filename,save_as), dpi=fig_dpi, bbox_inches='tight')
        try:    plt.close(fig)
        except: pass
        return fig

def plot_KDE(sol, var1, var2, fig=None, ax=None, save=False, save_as_png=True, fig_dpi=144):
    if True:
        save_as = 'png'
    else:
        save_as = 'pdf'
    if var1 == var2:
        fig, ax = plt.subplots(figsize=(3,3))
        plt.close(fig)
        return fig
    else:
        if fig == None or ax == None:
            fig, ax = plt.subplots(figsize=(3,3))
        MDL = sol.MDL
        filename = sol.filename.replace("\\", "/").split("/")[-1].split(".")[0]
        model = sol.get_model_type()
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
        plt.axes(ax)
        # Contourf plot
        plt.grid(None)
        plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
        ax.scatter(x, y, color='k', s=1)
        plt.xticks(rotation=90)
        plt.locator_params(axis = 'y', nbins = 7)
        plt.locator_params(axis = 'x', nbins = 7)
        cfset = ax.contourf(xx, yy, f, cmap=plt.cm.Blues, alpha=0.7)
        ## Or kernel density estimate plot instead of the contourf plot
#        ax.imshow(np.rot90(f), cmap='Blues', extent=[xmin, xmax, ymin, ymax])
        # Contour plot
#        cset = ax.contour(xx, yy, f, levels=cfset.levels[2::2], colors='k', alpha=0.8)
        # Label plot
    #    ax.clabel(cset, cset.levels[::1], inline=1, fmt='%.1E', fontsize=10)
        plt.yticks(fontsize=14)
        plt.xticks(fontsize=14)
        plt.ylabel("%s" %var2, fontsize=14)
        plt.xlabel("%s" %var1, fontsize=14)
    #    plt.gca().get_xaxis().get_major_formatter().set_useOffset(False)
    #    plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)
        if save:
            save_where = '/Figures/Bivariate KDE/%s/' %filename
            working_path = getcwd().replace("\\", "/")+"/"
            save_path = working_path+save_where
            print("\nSaving KDE figure in:\n", save_path)
            if not path.exists(save_path):
                makedirs(save_path)
            fig.savefig(save_path+'KDE-%s-%s_%s_%s.%s'%(model,filename,var1,var2,save_as), dpi=fig_dpi, bbox_inches='tight')
        plt.close(fig)
        return fig

def plot_hexbin(sol, var1, var2, save=False, save_as_png=True, fig_dpi=144):
    if save_as_png:
        save_as = 'png'
    else:
        save_as = 'pdf'
    MDL = sol.MDL
    filename = sol.filename.replace("\\", "/").split("/")[-1].split(".")[0]
    model = sol.get_model_type()
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
    fig, ax = plt.subplots(figsize=(4,4))
    plt.grid(None)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
#    plt.scatter(x, y)
    plt.hexbin(x, y, gridsize=20, cmap=plt.cm.Blues)
    plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
    plt.xticks(rotation=90)
    plt.locator_params(axis = 'y', nbins = 5)
    plt.locator_params(axis = 'x', nbins = 5)    
    cb = plt.colorbar()
    cb.set_label('Number of observations')
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.ylabel("%s" %var2, fontsize=14)
    plt.xlabel("%s" %var1, fontsize=14)
    if save:
        save_where = '/Figures/Hexbins/%s/' %filename
        working_path = getcwd().replace("\\", "/")+"/"
        save_path = working_path+save_where
        print("\nSaving hexbin figure in:\n", save_path)
        if not path.exists(save_path):
            makedirs(save_path)
        fig.savefig(save_path+'Bivar-%s-%s_%s_%s.%s'%(model,filename,var1,var2,save_as), dpi=fig_dpi, bbox_inches='tight')
    plt.close(fig)
    return fig

def plot_traces(sol, no_subplots=False, save=False, save_as_png=True, fig_dpi=144):
    if save_as_png:
        save_as = 'png'
    else:
        save_as = 'pdf'
    MDL = sol.MDL
    model = get_model_type(sol)
    filename = sol.filename.replace("\\", "/").split("/")[-1].split(".")[0]
    keys = sorted([x.__name__ for x in MDL.deterministics]) + sorted([x.__name__ for x in MDL.stochastics])
#    keys = sorted([x.__name__ for x in MDL.stochastics])
    sampler = MDL.get_state()["sampler"]
    try:
        keys.remove("zmod")
        keys.remove("log_m_i")
        keys.remove("log_tau_i")
        keys.remove("cond")
    except:
        pass
    for (i, k) in enumerate(keys):
        vect = old_div((MDL.trace(k)[:].size),(len(MDL.trace(k)[:])))
        if vect > 1:
            if "Decomp" in model:
                keys[i] = [k+"%d"%n for n in range(0,vect)]
            else:
                keys[i] = [k+"%d"%n for n in range(1,vect+1)]
    keys = list(flatten(keys))
    ncols = 2
    nrows = int(ceil(len(keys)*1.0 / ncols))
    if no_subplots:
        figs = {}
        for c, k in enumerate(keys):
            fig, ax = plt.subplots(figsize=(8,4))
            plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
            if k == "R0":
                stoc = "R0"
            else:
                stoc =  ''.join([i for i in k if not i.isdigit()])
                stoc_num = [int(i) for i in k if i.isdigit()]
            try:
                data = MDL.trace(stoc)[:][:,stoc_num[0]-1]
            except:
                data = MDL.trace(stoc)[:]
            x = np.arange(sampler["_burn"]+1, sampler["_iter"]+1, sampler["_thin"])
            plt.ylabel("%s value" %k)
            plt.xlabel("Iteration number")
            
            if sampler["_burn"] == 0:
                plt.xscale('log')
            else:
                plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
                
            plt.plot(x, data, '-', label=k)
            plt.grid('on')
            if save:
                save_where = '/Figures/Traces/%s/' %filename
                working_path = getcwd().replace("\\", "/")+"/"
                save_path = working_path+save_where
                if c == 0:
                    print("\nSaving traces figure in:\n", save_path)
                if not path.exists(save_path):
                    makedirs(save_path)
                fig.savefig(save_path+'Trace-%s-%s-%s.%s'%(model,filename,k,save_as))
            figs[k] = fig
            plt.close(fig)
        return figs

    else:
        fig, ax = plt.subplots(nrows, ncols, figsize=(8,nrows*1.4))
        for c, (a, k) in enumerate(zip(ax.flat, keys)):
            if k == "R0":
                stoc = "R0"
            else:
                stoc =  ''.join([i for i in k if not i.isdigit()])
                stoc_num = [int(i) for i in k if i.isdigit()]
            try:
                data = MDL.trace(stoc)[:][:,stoc_num[0]-1]
            except:
                data = MDL.trace(stoc)[:]
            x = np.arange(sampler["_burn"]+1, sampler["_iter"]+1, sampler["_thin"])
            plt.axes(a)
            plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
            plt.locator_params(axis = 'y', nbins = 6)
            plt.ylabel(k)
            plt.plot(x, data, '-')
#            plt.xscale('log')
            plt.grid('on')
            
        plt.tight_layout(pad=0.1, w_pad=0., h_pad=-2)
        for a in ax.flat[ax.size - 1:len(keys) - 1:-1]:
            a.set_visible(False)
        
        ax[:,0][-1].set_xlabel("Iteration number")
        for a in ax[:-1]:
            a[0].axes.get_xaxis().set_ticklabels([])
        
        if len(keys) % 2 == 0:
            ax[:,1][-1].set_xlabel("Iteration number")
            for a in ax[:-1]:
                a[1].axes.get_xaxis().set_ticklabels([])
        else:
            ax[:,1][-2].set_xlabel("Iteration number")
            for a in ax[:-2]:    
                a[1].axes.get_xaxis().set_ticklabels([])
            
        if save:
            save_where = '/Figures/Traces/'
            working_path = getcwd().replace("\\", "/")+"/"
            save_path = working_path+save_where
            print("\nSaving trace figures in:\n", save_path)
            if not path.exists(save_path):
                makedirs(save_path)
            fig.savefig(save_path+'Trace-%s-%s.%s'%(model,filename,save_as), dpi=fig_dpi, bbox_inches='tight')
        plt.close(fig)
        return fig

def plot_summary(sol, save=False, save_as_png=True, fig_dpi=144):
    if save_as_png:
        save_as = 'png'
    else:
        save_as = 'pdf'
    MDL, ch_n = sol.MDL, sol.mcmc["nb_chain"]
    model = get_model_type(sol)
    filename = sol.filename.replace("\\", "/").split("/")[-1].split(".")[0]
    keys = sorted([x.__name__ for x in MDL.deterministics]) + sorted([x.__name__ for x in MDL.stochastics])
    try:
        keys.remove("zmod")
        keys.remove("log_m_i")
        keys.remove("log_tau_i")
        keys.remove("cond")
    except:
        pass
    for (i, k) in enumerate(keys):
        vect = old_div((MDL.trace(k)[:].size),(len(MDL.trace(k)[:])))
        if vect > 1:
         keys[i] = [k+"%d"%n for n in range(1,vect+1)]
    keys = list(reversed(sorted(flatten(keys))))
    try:    r_hat = gelman_rubin(MDL)
    except:
        print("\nTwo or more chains of equal length required for Gelman-Rubin convergence")
    fig, axes = plt.subplots(figsize=(6,4))
    gs2 = gridspec.GridSpec(3, 3)
    ax1 = plt.subplot(gs2[:, :-1])
    ax2 = plt.subplot(gs2[:, -1], sharey = ax1)
    ax2.set_xlabel("R-hat")
    ax2.plot([1,1], [-1,len(keys)], "--", color="C7", zorder=0)
    for (i, k) in enumerate(keys):
        test = k[-1] not in ["%d"%d for d in range(1,8)] or k == "R0"
        for c in range(ch_n):
            if test:
                imp = None
                val_m = MDL.stats(k[:imp], chain=c)[k[:imp]]['mean']
                hpd_h = MDL.stats(k[:imp], chain=c)[k[:imp]]['95% HPD interval'][0]
                hpd_l = MDL.stats(k[:imp], chain=c)[k[:imp]]['95% HPD interval'][1]
            else:
                imp = -1
                val_m = MDL.stats(k[:imp], chain=c)[k[:imp]]['mean'][int(k[-1])-1]
                hpd_h = MDL.stats(k[:imp], chain=c)[k[:imp]]['95% HPD interval'][0][int(k[-1])-1]
                hpd_l = MDL.stats(k[:imp], chain=c)[k[:imp]]['95% HPD interval'][1][int(k[-1])-1]
            val = val_m
            err = [[abs(hpd_h-val_m)],
                    [abs(hpd_l-val_m)]]
            if ch_n % 2 != 0:   o_s = 0
            else:               o_s = 0.5
            ax1.scatter(val, i - (old_div(ch_n,2))*(1./ch_n/1.4) + (1./ch_n/1.4)*(c+o_s), color="C0", marker="o", s=50, edgecolors='C7',alpha=0.7)
            ax1.errorbar(val, i - (old_div(ch_n,2))*(1./ch_n/1.4) + (1./ch_n/1.4)*(c+o_s), xerr=err, color="C7", fmt=" ", zorder=0)
        if ch_n >= 2:
            R = np.array(r_hat[k[:imp]])
            R[R > 3] = 3
            if test:
                ax2.scatter(R, i, color="C1", marker="<", s=50, alpha=0.7)
            else:
                ax2.scatter(R[int(k[-1])-1], i, color="C1", marker="<", s=50, alpha=0.7)
    
    ax1.set_ylim([-1, len(keys)])
    ax1.set_yticks(list(range(0,len(keys))))
    ax1.set_yticklabels(keys)
    plt.setp(ax2.get_yticklabels(), visible=False)
    ax2.set_xlim([0.5, 3.5])
    ax2.set_xticklabels(["","1","2","3+"])
    ax2.set_xticks([0.5, 1, 2, 3])
    ax1.set_xlabel("Parameter values")
    plt.tight_layout()
    
    if save:
        save_where = '/Figures/Summaries/'
        working_path = getcwd().replace("\\", "/")+"/"
        save_path = working_path+save_where
        print("\nSaving summary figure in:\n", save_path)
        if not path.exists(save_path):
            makedirs(save_path)
        fig.savefig(save_path+'Summary-%s-%s.%s'%(model,filename,save_as), dpi=fig_dpi, bbox_inches='tight')
    try:    plt.close(fig)
    except: pass

    return fig

def plot_autocorr(sol, save=False, save_as_png=True, fig_dpi=144):
    if save_as_png:
        save_as = 'png'
    else:
        save_as = 'pdf'
    MDL = sol.MDL
    model = get_model_type(sol)
    filename = sol.filename.replace("\\", "/").split("/")[-1].split(".")[0]
    keys = sorted([x.__name__ for x in MDL.deterministics]) + sorted([x.__name__ for x in MDL.stochastics])
    try:
        keys.remove("zmod")
        keys.remove("log_m_i")
        keys.remove("log_tau_i")
        keys.remove("cond")    
    except:
        pass
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
        plt.axes(a)
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
        save_where = '/Figures/Autocorrelations/'
        working_path = getcwd().replace("\\", "/")+"/"
        save_path = working_path+save_where
        print("\nSaving autocorrelation figure in:\n", save_path)
        if not path.exists(save_path):
            makedirs(save_path)
        fig.savefig(save_path+'Autocorr-%s-%s.%s'%(model,filename,save_as), dpi=fig_dpi, bbox_inches='tight')
    try:    plt.close(fig)
    except: pass
    return fig

def plot_rtd(sol, save=False, draw=True, save_as_png=True, fig_dpi=144):
    if save_as_png:
        save_as = 'png'
    else:
        save_as = 'pdf'
    filename = sol.filename.replace("\\", "/").split("/")[-1].split(".")[0]
    model = get_model_type(sol)
    if draw or save:
        fig, ax = plt.subplots(figsize=(5,3))
        uncer_m = sol.MDL.stats()["log_m_i"]["standard deviation"]
        uncer_tau = sol.MDL.stats()["log_tau_i"]["standard deviation"]
        
        peaks = np.atleast_1d(sol.MDL.stats()["log_peak_tau"]["mean"])
        uncer_peaks = sol.MDL.stats()["log_peak_tau"]['95% HPD interval']
        uncer_peaks = uncer_peaks[1]-uncer_peaks[0]
        
        bot95 = sol.MDL.stats()["log_m_i"]['95% HPD interval'][0]
        top95 = sol.MDL.stats()["log_m_i"]['95% HPD interval'][1]
        
        log_tau = sol.MDL.stats()["log_tau_i"]['mean']
        log_m = sol.MDL.stats()["log_m_i"]['mean']
        m_peaks = log_m[[list(log_tau).index(peaks[x]) for x in range(len(peaks))]]
        plt.errorbar(log_tau, log_m, None, None, color="C7", linestyle='-', label="RTD (95% HPD)")
        plt.errorbar(peaks, m_peaks+0.1, None, uncer_peaks, color="C3", marker="v", linestyle="", label=r"$\tau_{peak}$ (95% HPD)")
        plt.fill_between(np.log10(sol.ccd_priors['tau']), bot95, top95, color="C7", alpha=0.2)
        plt.axvline(sol.MDL.stats()["log_mean_tau"]['mean'],color="#2ca02c",label=r"$\bar{\tau}$ (95% HPD)")
        plt.axvline(sol.MDL.stats()["log_half_tau"]['mean'],color='#1f77b4',label=r"$\tau_{50}$ (95% HPD)")
        inter = sol.MDL.stats()["log_mean_tau"]['95% HPD interval']
        plt.axvspan(inter[0], inter[1], alpha=0.2, color="#2ca02c")
        inter = sol.MDL.stats()["log_half_tau"]['95% HPD interval']
        plt.axvspan(inter[0], inter[1], alpha=0.2, color='#1f77b4')
        plt.axvspan(min(log_tau), min(log_tau)+1, alpha=0.1, color='C7')
        plt.axvspan(max(log_tau)-1, max(log_tau), alpha=0.1, color='C7')
        plt.xlim([min(log_tau), max(log_tau)])
        plt.xlabel(r"log$_{10}\tau$ ($\tau$ in s)", fontsize=12)
        plt.ylabel("log$_{10}$m", fontsize=12)
        plt.legend(fontsize=8, loc=1)
        plt.xlabel(r"$log_{10}\tau$ ($\tau$ in s)", fontsize=12)
        plt.ylabel(r"$log_{10}$m", fontsize=12)
        plt.yticks(fontsize=12), plt.xticks(fontsize=12)
        plt.grid('on')
        plt.legend(numpoints=1, fontsize=10, loc="best",labelspacing=0.1)
        fig.tight_layout()
    if save:
        save_where = '/Figures/Debye distributions/'
        working_path = getcwd().replace("\\", "/")+"/"

        save_path = working_path+save_where
        print("\nSaving relaxation time distribution figure in:\n", save_path)
        if not path.exists(save_path):
            makedirs(save_path)
        fig.savefig(save_path+'RTD-%s-%s.%s'%(model, filename,save_as), dpi=fig_dpi, bbox_inches='tight')
    try:    plt.close(fig)
    except: pass
    if draw:    return fig
    else:       return None

def save_resul(sol):
    # Fonction pour enregistrer les résultats
    MDL, pm, filepath = sol.MDL, sol.pm, sol.filename
    model = get_model_type(sol)
    print(model)
    sample_name = filepath.replace("\\", "/").split("/")[-1].split(".")[0]
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
    headers = []
    for c, key in enumerate(sorted(pm.keys())):
        A.append(list(np.array(pm[key]).ravel()))
        key = model+"_"+key
        if len(A[c]) == 1:
            headers.append(key)
#            print(key)
        else:
            for i in range(len(A[c])):
#                print(key+"%d" %(i+tag))
                headers.append(key.strip("_")+"%d" %(i+tag))
    headers = ','.join(headers)
    results = np.array(flatten(A))
    if sol.model == 'PDecomp': 
        tau_ = sol.data["tau"]
        add = ["tau"+"%d"%(i) for i in range(len(tau_))]
        add = ',' + ','.join(add)
        headers += add
        results = np.concatenate((results,tau_))
    headers += ",Z_max,c_exponent"
    results = np.concatenate((results,np.array([sol.data["Z_max"]]),np.array([sol.c_exp])))
    np.savetxt(save_path+'INV_%s_%s.csv' %(model,sample_name), results[None],
               header=headers, comments='', delimiter=',')
    vars_ = ["%s"%x for x in MDL.stochastics]+["%s"%x for x in MDL.deterministics]
    if "zmod" in vars_: vars_.remove("zmod")
    MDL.write_csv(save_path+'STATS_%s_%s.csv' %(model,sample_name), variables=(vars_))

def merge_results(sol,files):
    model = get_model_type(sol)
    save_where = '/Batch results/'
    working_path = getcwd().replace("\\", "/")+"/"
    save_path = working_path+save_where
    print("\nMerging csv files")
    if not path.exists(save_path):
        makedirs(save_path)
    to_merge = working_path+"/Results/%s/INV_%s_%s.csv" %(files[0],model,files[0])
    headers = np.genfromtxt(to_merge, delimiter=",", dtype=str, skip_footer=1)
    merged_inv_results = np.empty((len(files), len(headers)))
    for i, f in enumerate(files):
        merged_inv_results[i] = np.loadtxt(working_path+"/Results/%s/INV_%s_%s.csv" %(f,model,f), delimiter=",", skiprows=1)
    rows = np.array(files, dtype=str)[:, np.newaxis]
    hd = ",".join(["ID"] + list(headers))
    np.savetxt(save_path+"Merged_%s_%s_TO_%s.csv" %(model,files[0],files[-1]), np.hstack((rows, merged_inv_results)), delimiter=",", header=hd, fmt="%s")
    print("Batch file successfully saved in:\n", save_path)

def plot_data(filename, headers, ph_units):
    data = get_data(filename,headers,ph_units)
    # Graphiques du data
    Z = data["Z"]
    dZ = data["Z_err"]
    f = data["freq"]
    Zr0 = max(abs(Z))
    zn_dat = old_div(Z,Zr0)
    zn_err = old_div(dZ,Zr0)
    Pha_dat = 1000*data["pha"]
    Pha_err = 1000*data["pha_err"]
    Amp_dat = old_div(data["amp"],Zr0)
    Amp_err = old_div(data["amp_err"],Zr0)

    fig, ax = plt.subplots(3, 1, figsize=(6,8))
    for t in ax:
        t.tick_params(labelsize=12)
    # Real-Imag
    plt.axes(ax[0])
    plt.errorbar(zn_dat.real, -zn_dat.imag, zn_err.imag, zn_err.real, '.b', label=filename)
    plt.xlabel(sym_labels['real'], fontsize=12)
    plt.ylabel(sym_labels['imag'], fontsize=12)

    plt.xlim([None, 1])
    plt.ylim([0, None])
#    plt.legend(numpoints=1, fontsize=9)
#    plt.title(filename, fontsize=10)
    # Freq-Phas
    plt.axes(ax[1])
    plt.errorbar(f, -Pha_dat, Pha_err, None, '.b', label=filename)
    ax[1].set_yscale("log", nonposy='clip')
    ax[1].set_xscale("log")
    plt.xlabel(sym_labels['freq'], fontsize=12)
    plt.ylabel(sym_labels['phas'], fontsize=12)
#    plt.legend(loc=2, numpoints=1, fontsize=9)
    plt.ylim([1,1000])
    # Freq-Ampl
    plt.axes(ax[2])
    plt.errorbar(f, Amp_dat, Amp_err, None, '.b', label=filename)
    ax[2].set_xscale("log")
    plt.xlabel(sym_labels['freq'], fontsize=12)
    plt.ylabel(sym_labels['ampl'], fontsize=12)
    plt.ylim([None,1.0])
#    plt.legend(numpoints=1, fontsize=9)
    fig.tight_layout()

    plt.close(fig)
    return fig

def plot_deviance(sol, save=False, draw=True, save_as_png=True, fig_dpi=144):
    if save_as_png:
        save_as = 'png'
    else:
        save_as = 'pdf'
    filename = sol.filename.replace("\\", "/").split("/")[-1].split(".")[0]
    model = get_model_type(sol)
    if draw or save:
        fig, ax = plt.subplots(figsize=(4,3))
        deviance = sol.MDL.trace('deviance')[:]
        sampler_state = sol.MDL.get_state()["sampler"]
        x = np.arange(sampler_state["_burn"]+1, sampler_state["_iter"]+1, sampler_state["_thin"])
        plt.plot(x, deviance, "-", color="C3", label="Model deviance\nDIC = %.2f\nBPIC = %.2f" %(sol.MDL.DIC,sol.MDL.BPIC))
        plt.xlabel("Iteration")
        plt.ylabel("Deviance")
        plt.legend(numpoints=1, loc="best")
        if sampler_state["_burn"] == 0:
            plt.xscale('log')
        else:
            plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        fig.tight_layout()
    if save:
        save_where = '/Figures/ModelDeviance/'
        working_path = getcwd().replace("\\", "/")+"/"
        save_path = working_path+save_where
        print("\nSaving model deviance figure in:\n", save_path)
        if not path.exists(save_path):
            makedirs(save_path)
        fig.savefig(save_path+'ModelDeviance-%s-%s.%s'%(model,filename,save_as), dpi=fig_dpi, bbox_inches='tight')
    try:    plt.close(fig)
    except: pass
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

def plot_logp(sol, save=False, draw=True, save_as_png=True, fig_dpi=144):
    if save_as_png:
        save_as = 'png'
    else:
        save_as = 'pdf'
    filename = sol.filename.replace("\\", "/").split("/")[-1].split(".")[0]
    model = get_model_type(sol)
    if draw or save:
        fig, ax = plt.subplots(figsize=(4,3))
        logp = logp_trace(sol.MDL)
        sampler_state = sol.MDL.get_state()["sampler"]
        x = np.arange(sampler_state["_burn"]+1, sampler_state["_iter"]+1, sampler_state["_thin"])
        plt.plot(x, logp, "-", color="C3")
        plt.xlabel("Iteration")
        plt.ylabel("Log-likelihood")
        plt.legend(numpoints=1, loc="best")
        if sampler_state["_burn"] == 0:
            plt.xscale('log')
        else:
            plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        fig.tight_layout()
    if save:
        save_where = '/Figures/LogLikelihood/'
        working_path = getcwd().replace("\\", "/")+"/"
        save_path = working_path+save_where
        print("\nSaving logp trace figure in:\n", save_path)
        if not path.exists(save_path):
            makedirs(save_path)
        fig.savefig(save_path+'LogLikelihood-%s-%s.%s'%(model,filename,save_as), dpi=fig_dpi, bbox_inches='tight')
    try:    plt.close(fig)
    except: pass
    if draw:    return fig
    else:       return None

def plot_fit(sol, save=False, draw=True, save_as_png=True, fig_dpi=144):
    if save_as_png:
        save_as = 'png'
    else:
        save_as = 'pdf'
    filepath = sol.filename
    sample_name = filepath.replace("\\", "/").split("/")[-1].split(".")[0]
    model = get_model_type(sol)
    data = sol.data
    fit = sol.fit
    # Graphiques du fit
    f = data["freq"]
    Zr0 = max(abs(data["Z"]))
    zn_dat = old_div(data["Z"],Zr0)
    zn_err = old_div(data["Z_err"],Zr0)
    zn_fit = old_div(fit["best"],Zr0)
    zn_min = old_div(fit["lo95"],Zr0)
    zn_max = old_div(fit["up95"],Zr0)
    Pha_dat = 1000*data["pha"]
    Pha_err = 1000*data["pha_err"]
    Pha_fit = 1000*np.angle(fit["best"])
    Pha_min = 1000*np.angle(fit["lo95"])
    Pha_max = 1000*np.angle(fit["up95"])
    Amp_dat = old_div(data["amp"],Zr0)
    Amp_err = old_div(data["amp_err"],Zr0)
    Amp_fit = old_div(abs(fit["best"]),Zr0)
    Amp_min = old_div(abs(fit["lo95"]),Zr0)
    Amp_max = old_div(abs(fit["up95"]),Zr0)
    if draw or save:
        fig, ax = plt.subplots(1, 3, figsize=(11,3))
#        for t in ax:
#            t.tick_params(labelsize=14)
        # Real-Imag
        plt.axes(ax[2])
        plt.errorbar(zn_fit.real, -zn_dat.imag, zn_err.imag, zn_err.real, '.', color="#1f77b4", label='Data')
        plt.plot(zn_fit.real, -zn_fit.imag, '-', color="#ff7f0e", label='Model')
        plt.fill_between(zn_fit.real, -zn_max.imag, -zn_min.imag, color="#ff7f0e", alpha=0.2, label="95% HPD")
        plt.xlabel(sym_labels['real'])
        plt.ylabel(sym_labels['imag'])
        plt.legend(loc='best', fontsize=9, numpoints=1)
        plt.xlim([None, 1])
        plt.ylim([0, max(-zn_dat.imag)])
        
        # Freq-Ampl
        plt.axes(ax[1])
        plt.errorbar(f, Amp_dat, Amp_err, None, '.', color="#1f77b4", label='Data')
        plt.semilogx(f, Amp_fit, '-', color="#ff7f0e", label='Model')
        plt.fill_between(f, Amp_max, Amp_min, color="#ff7f0e", alpha=0.2, label="95% HPD")
        plt.xlabel(sym_labels['freq'])
        plt.ylabel(sym_labels['ampl'])
        plt.legend(loc='best', fontsize=9, numpoints=1)
        plt.xlim([10**np.floor(min(np.log10(f))), 10**np.ceil(max(np.log10(f)))])
        plt.ylim([None,1.0])

        # Freq-Phas
        plt.axes(ax[0])
        plt.errorbar(f, -Pha_dat, Pha_err, None, '.', color="#1f77b4", label='Data')
        plt.loglog(f, -Pha_fit, '-', color="#ff7f0e", label='Model')
        ax[0].set_yscale("log", nonposy='clip')
        plt.fill_between(f, -Pha_max, -Pha_min, color="#ff7f0e", alpha=0.2, label="95% HPD")
        plt.xlabel(sym_labels['freq'])
        plt.ylabel(sym_labels['phas'])
        plt.legend(loc='best', fontsize=9, numpoints=1)
        plt.xlim([10**np.floor(min(np.log10(f))), 10**np.ceil(max(np.log10(f)))])
        plt.ylim([1,10**np.ceil(max(np.log10(-Pha_dat)))])

        for a in ax:
            a.grid('on')

        if  (-Pha_dat < 1).any() and (-Pha_dat >= 0.1).any():
            plt.ylim([0.1,10**np.ceil(max(np.log10(-Pha_dat)))])  
        if  (-Pha_dat < 0.1).any() and (-Pha_dat >= 0.01).any():
            plt.ylim([0.01,10**np.ceil(max(np.log10(-Pha_dat)))]) 
        
        plt.tight_layout(pad=1, h_pad=0, w_pad=0.5)
        
    if save:
        save_where = '/Figures/Fit figures/'
        working_path = getcwd().replace("\\", "/")+"/"
        save_path = working_path+save_where
        print("\nSaving fit figure in:\n", save_path)
        if not path.exists(save_path):
            makedirs(save_path)
        fig.savefig(save_path+'FIT-%s-%s.%s'%(model,sample_name,save_as), dpi=fig_dpi, bbox_inches='tight')
    try:    plt.close(fig)
    except: pass
    if draw:    return fig
    else:       return None

def print_diagn(M, q, r, s):
    return raftery_lewis(M, q, r, s, verbose=0)

def plot_par():
#    rc = rcParams.items()    
    rc = {u'figure.dpi': 72.0,
          u'figure.edgecolor': 'white',
          u'figure.facecolor': 'white',
          u'savefig.bbox': u'tight',
          u'savefig.directory': u'~',
          u'savefig.dpi': 200.0,
          u'savefig.edgecolor': u'white',
          u'savefig.facecolor': u'white',
          u'axes.formatter.use_mathtext': True,
          u'mathtext.default' : 'regular',
          }
    return rc
rcParams.update(plot_par())
