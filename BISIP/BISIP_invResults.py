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
import matplotlib.ticker as mticker
import numpy as np
from os import path, makedirs
from sys import argv
from math import ceil
from pymc import raftery_lewis, gelman_rubin, geweke
from scipy.stats import norm, gaussian_kde
from BISIP_models import get_data
#==============================================================================

sym_labels = dict([('resi', r"$\rho\/(\Omega\cdot m)$"),
                   ('freq', r"Frequency $(Hz)$"),
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
    model = sol["SIP_model"]
    if model == "PDecomp":
        if sol["model_type"]["c_exp"] == 0.5:
            model = "WarburgDecomp"
        elif sol["model_type"]["c_exp"] == 1.0:
            model = "DebyeDecomp"
        else:
            model = "ColeColeDecomp"
    return model

def print_resul(sol):
#==============================================================================
    # Impression des résultats
    pm, model, filename = sol["params"], sol["SIP_model"], sol["path"]
    print('\n\nInversion success!')
    print('Name of file:', filename)
    print('Model used:', model)
    e_keys = sorted([s for s in list(pm.keys()) if "_std" in s])
    v_keys = [e.replace("_std", "") for e in e_keys]
    labels = ["{:<8}".format(x+":") for x in v_keys]
    np.set_printoptions(formatter={'float': lambda x: format(x, '6.3E')})
    for l, v, e in zip(labels, v_keys, e_keys):
        print(l, pm[v], '+/-', pm[e], np.char.mod('(%.2f%%)',abs(100*pm[e]/pm[v])))

def plot_histo(sol, no_subplots=False, save=False, save_as_png=True):
    if save_as_png:
        save_as = 'png'
    else:
        save_as = 'pdf'
    MDL = sol["pymc_model"]
    model = get_model_type(sol)
    filename = sol["path"].replace("\\", "/").split("/")[-1].split(".")[0]
    keys = sorted([x.__name__ for x in MDL.deterministics]) + sorted([x.__name__ for x in MDL.stochastics])
    try:
        keys.remove("zmod")
        keys.remove("m_")
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
            fit = norm.pdf(data, np.mean(data), np.std(data))
            plt.xlabel("%s value"%k)
            plt.ylabel("Probability density")
            hist = plt.hist(data, bins=20, normed=True, linewidth=1.0, color="white")
            plt.plot(data, fit, "-", label="Fitted PDF", linewidth=1.5)
            plt.legend(loc='best')
            plt.grid('off')
            if save:
                save_where = '/Figures/Histograms/%s/' %filename
                actual_path = str(path.dirname(path.realpath(argv[0]))).replace("\\", "/")
                save_path = actual_path+save_where
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
            fit = norm.pdf(data, np.mean(data), np.std(data))
            plt.axes(a)
            plt.locator_params(axis = 'y', nbins = 8)
            plt.locator_params(axis = 'x', nbins = 7)
            plt.xlabel(k)
            hist = plt.hist(data, bins=20, normed=False, label=filename, edgecolor='black', linewidth=1.0, color="white")
            xh = [0.5 * (hist[1][r] + hist[1][r+1]) for r in range(len(hist[1])-1)]
            binwidth = old_div((max(xh) - min(xh)), len(hist[1]))
            fit *= len(data) * binwidth
            plt.plot(data, fit, "-", linewidth=1.5)
            plt.grid('off')
            plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
        
        for c in range(nrows):
            ax[c][0].set_ylabel("Frequency")

        plt.tight_layout(pad=1, w_pad=1, h_pad=0)
        for a in ax.flat[ax.size - 1:len(keys) - 1:-1]:
            a.set_visible(False)
        if save:
            save_where = '/Figures/Histograms/'
            actual_path = str(path.dirname(path.realpath(argv[0]))).replace("\\", "/")
            save_path = actual_path+save_where
            print("\nSaving parameter histograms in:\n", save_path)
            if not path.exists(save_path):
                makedirs(save_path)
            fig.savefig(save_path+'Histo-%s-%s.%s'%(model,filename,save_as), bbox_inches='tight')
        try:    plt.close(fig)
        except: pass
        return fig

def plot_KDE(sol, var1, var2, fig=None, ax=None, save=False, save_as_png=True):
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
        MDL = sol["pymc_model"]
        filename = sol["path"].replace("\\", "/").split("/")[-1].split(".")[0]
        model = get_model_type(sol)
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
        kernel.set_bandwidth(bw_method=kernel.factor * 2.)
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
            actual_path = str(path.dirname(path.realpath(argv[0]))).replace("\\", "/")
            save_path = actual_path+save_where
            print("\nSaving KDE figure in:\n", save_path)
            if not path.exists(save_path):
                makedirs(save_path)
            fig.savefig(save_path+'KDE-%s-%s_%s_%s.%s'%(model,filename,var1,var2,save_as), dpi=200)
        plt.close(fig)
        return fig

def plot_all_KDE(sol):
    MDL = sol["pymc_model"]
    model = get_model_type(sol)
    filename = sol["path"].replace("\\", "/").split("/")[-1].split(".")[0]
    keys = sorted([x.__name__ for x in MDL.deterministics]) + sorted([x.__name__ for x in MDL.stochastics])
    sampler = MDL.get_state()["sampler"]
    try:
        keys.remove("zmod")
        keys.remove("m_")
        keys.remove("log_mean_tau")
        keys.remove("total_m")

    except:
        pass
    for (i, k) in enumerate(keys):
        vect = old_div((MDL.trace(k)[:].size),(len(MDL.trace(k)[:])))
        if vect > 1:
            keys[i] = [k+"%d"%n for n in range(0,vect)]
    keys = list(flatten(keys))
    ncols = len(keys)
    nrows = len(keys)
#    fig, ax = plt.subplots(nrows, ncols, figsize=(10,10))
    
    fig = plt.figure(figsize=(10,10))
#    plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
    plotz = len(keys)    
    for i in range(plotz):
        for j in range(plotz):
            if j<i:
                var1 = keys[j]
                var2 = keys[i]
                print((var1, var2))
                ax = plt.subplot2grid((plotz-1, plotz-1), (i-1,j))
                ax.ticklabel_format(axis='y', style='sci', scilimits=(0,1))
                ax.ticklabel_format(axis='x', style='sci', scilimits=(0,1))
                
                if var1 == "R0":
                    stoc1 = "R0"
                else:
                    stoc1 =  ''.join([k for k in var1 if not k.isdigit()])
                    stoc_num1 = [int(k) for k in var1 if k.isdigit()]
                try:
                    x = MDL.trace(stoc1)[:,stoc_num1[0]-1]
                except:
                    x = MDL.trace(stoc1)[:]
                if var2 == "R0":
                    stoc2 = "R0"
                else:
                    stoc2 =  ''.join([k for k in var2 if not k.isdigit()])
                    stoc_num2 = [int(k) for k in var2 if k.isdigit()]
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
                kernel.set_bandwidth(bw_method=kernel.factor * 2.)
                f = np.reshape(kernel(positions).T, xx.shape)
            
                ax.set_xlim(xmin, xmax)
                ax.set_ylim(ymin, ymax)
                plt.axes(ax)
                # Contourf plot
                plt.grid(None)
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
                if j == 0:
                    plt.ylabel("%s" %var2, fontsize=14)
                if i == len(keys)-1:
                    plt.xlabel("%s" %var1, fontsize=14)
                if j != 0:
                    ax.yaxis.set_ticklabels([])
                if i != len(keys)-1:
                    ax.xaxis.set_ticklabels([])
                plt.suptitle("Adaptive Metropolis step method", fontsize=14)
    
    
    
    
#    for v1 in range(len(keys)):
#        for v2 in range(len(keys)):
#            if v1 < v2:
#            
##                if v1 == v2:
##                    if keys[v1] == "R0":
##                        stoc1 = "R0"
##                    else:
##                        stoc1 =  ''.join([i for i in keys[v1] if not i.isdigit()])
##                        stoc_num1 = [int(i) for i in keys[v1] if i.isdigit()]
##                    try:
##                        x = MDL.trace(stoc1)[:,stoc_num1[0]-1]
##                    except:
##                        x = MDL.trace(stoc1)[:]
##                        ax[v1,v2].hist(x)
#                
#                fig = plot_KDE(sol, keys[v1], keys[v2], fig, ax[v2, v1])
#                print v1, v2
##            ax[i,j] = axs
#            fig = plot_axes(axs, fig)
#            fig.axes.append(axs)

    fig.tight_layout(pad=0, w_pad=1.0, h_pad=0.1)         
    return fig

def plot_hexbin(sol, var1, var2, save=False, save_as_png=True):
    if save_as_png:
        save_as = 'png'
    else:
        save_as = 'pdf'
    MDL = sol["pymc_model"]
    filename = sol["path"].replace("\\", "/").split("/")[-1].split(".")[0]
    model = get_model_type(sol)
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
        actual_path = str(path.dirname(path.realpath(argv[0]))).replace("\\", "/")
        save_path = actual_path+save_where
        print("\nSaving hexbin figure in:\n", save_path)
        if not path.exists(save_path):
            makedirs(save_path)
        fig.savefig(save_path+'Bivar-%s-%s_%s_%s.%s'%(model,filename,var1,var2,save_as))
    plt.close(fig)
    return fig

def plot_traces(sol, no_subplots=False, save=False, save_as_png=True):
    if save_as_png:
        save_as = 'png'
    else:
        save_as = 'pdf'
    MDL = sol["pymc_model"]
    model = get_model_type(sol)
    filename = sol["path"].replace("\\", "/").split("/")[-1].split(".")[0]
    keys = sorted([x.__name__ for x in MDL.deterministics]) + sorted([x.__name__ for x in MDL.stochastics])
    sampler = MDL.get_state()["sampler"]
    try:
        keys.remove("zmod")
        keys.remove("m_")
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
            plt.plot(x, data, '-', label=k, linewidth=2.0)
            if save:
                save_where = '/Figures/Traces/%s/' %filename
                actual_path = str(path.dirname(path.realpath(argv[0]))).replace("\\", "/")
                save_path = actual_path+save_where
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
            plt.plot(x, data, '-', label=filename, linewidth=1.0)
            plt.grid(None)
            
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
            actual_path = str(path.dirname(path.realpath(argv[0]))).replace("\\", "/")
            save_path = actual_path+save_where
            print("\nSaving trace figures in:\n", save_path)
            if not path.exists(save_path):
                makedirs(save_path)
            fig.savefig(save_path+'Trace-%s-%s.%s'%(model,filename,save_as))
        plt.close(fig)
        return fig

def plot_summary(sol, save=False, save_as_png=True):
    if save_as_png:
        save_as = 'png'
    else:
        save_as = 'pdf'
    MDL, ch_n = sol["pymc_model"], sol["mcmc"]["nb_chain"]
    model = get_model_type(sol)
    filename = sol["path"].replace("\\", "/").split("/")[-1].split(".")[0]
    keys = sorted([x.__name__ for x in MDL.deterministics]) + sorted([x.__name__ for x in MDL.stochastics])
    try:
        keys.remove("zmod")
        keys.remove("m_")
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
    fig, axes = plt.subplots(figsize=(8,5))
    gs2 = gridspec.GridSpec(3, 3)
    ax1 = plt.subplot(gs2[:, :-1])
    ax2 = plt.subplot(gs2[:, -1], sharey = ax1)
    ax2.set_xlabel("R-hat")
    ax2.plot([1,1], [-1,len(keys)], "--b")
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
            ax1.scatter(val, i - (old_div(ch_n,2))*(1./ch_n/1.4) + (1./ch_n/1.4)*(c+o_s), color="DeepSkyBlue", marker="o", s=50, edgecolors='k')
            ax1.errorbar(val, i - (old_div(ch_n,2))*(1./ch_n/1.4) + (1./ch_n/1.4)*(c+o_s), xerr=err, color="k", fmt=" ", zorder=0)
        if ch_n >= 2:
            R = np.array(r_hat[k[:imp]])
            R[R > 3] = 3
            if test:
                ax2.scatter(R, i, color="b", marker="s", s=50, edgecolors='k')
            else:
                ax2.scatter(R[int(k[-1])-1], i, color="b", marker="s", s=50, edgecolors='k')
    
    ax1.set_ylim([-1, len(keys)])
    ax1.set_yticks(list(range(0,len(keys))))
    ax1.set_yticklabels(keys)
    plt.setp(ax2.get_yticklabels(), visible=False)
    ax2.set_xlim([0.5, 3.5])
    ax2.set_xticklabels(["","1","2","3+"])
    ax2.set_xticks([0.5, 1, 2, 3])
    ax1.set_xlabel("Parameter values")

    if save:
        save_where = '/Figures/Summaries/'
        actual_path = str(path.dirname(path.realpath(argv[0]))).replace("\\", "/")
        save_path = actual_path+save_where
        print("\nSaving summary figure in:\n", save_path)
        if not path.exists(save_path):
            makedirs(save_path)
        fig.savefig(save_path+'Summary-%s-%s.%s'%(model,filename,save_as))
    try:    plt.close(fig)
    except: pass

    return fig

def plot_autocorr(sol, save=False, save_as_png=True):
    if save_as_png:
        save_as = 'png'
    else:
        save_as = 'pdf'
    MDL = sol["pymc_model"]
    model = get_model_type(sol)
    filename = sol["path"].replace("\\", "/").split("/")[-1].split(".")[0]
    keys = sorted([x.__name__ for x in MDL.deterministics]) + sorted([x.__name__ for x in MDL.stochastics])
    try:
        keys.remove("zmod")
        keys.remove("m_")
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
        actual_path = str(path.dirname(path.realpath(argv[0]))).replace("\\", "/")
        save_path = actual_path+save_where
        print("\nSaving autocorrelation figure in:\n", save_path)
        if not path.exists(save_path):
            makedirs(save_path)
        fig.savefig(save_path+'Autocorr-%s-%s.%s'%(model,filename,save_as))
    try:    plt.close(fig)
    except: pass
    return fig

def plot_debye(sol, save=False, draw=False, save_as_png=True):
    if save_as_png:
        save_as = 'png'
    else:
        save_as = 'pdf'
    filename = sol["path"].replace("\\", "/").split("/")[-1].split(".")[0]
    model = get_model_type(sol)
    if draw or save:
        fig, ax = plt.subplots(figsize=(6,4))
        x = np.log10(sol["data"]["tau"])
        xmax = max(x)-1
        xmin = min(x)+1
        x = np.linspace(xmin, xmax,100)
        y = 100*np.sum([a*(x**i) for (i, a) in enumerate(sol["params"]["a"])], axis=0)
        plt.errorbar(10**x, y, None, None, "-k", linewidth=2, label="Debye RTD")  
        ax.fill_between(10**x, 0, y, where=x >= -3, color='lightgray', alpha=0.7)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
        plt.xlabel("Relaxation time (s)", fontsize=14)
        plt.ylabel("Chargeability (%)", fontsize=14)
        plt.yticks(fontsize=14), plt.xticks(fontsize=14)
        plt.xscale("log")
        plt.xlim([10**xmin, 10**xmax])
        plt.ylim([0, None])
        plt.legend(numpoints=1, fontsize=14, loc="best")
        fig.tight_layout()
    if save:
        save_where = '/Figures/Debye distributions/'
        actual_path = str(path.dirname(path.realpath(argv[0]))).replace("\\", "/")
        save_path = actual_path+save_where
        print("\nSaving relaxation time distribution figure in:\n", save_path)
        if not path.exists(save_path):
            makedirs(save_path)
        fig.savefig(save_path+'Polynomial-RTD-%s-%s.%s'%(model, filename,save_as))
    try:    plt.close(fig)
    except: pass
    if draw:    return fig
    else:       return None

def save_resul(sol):
    # Fonction pour enregistrer les résultats
    MDL, pm, filepath = sol["pymc_model"], sol["params"], sol["path"]
    model = get_model_type(sol)
    sample_name = filepath.replace("\\", "/").split("/")[-1].split(".")[0]
    save_where = '/Results/'
    actual_path = str(path.dirname(path.realpath(argv[0]))).replace("\\", "/")
    save_path = actual_path+save_where+"%s/"%sample_name
    print("\nSaving csv file in:\n", save_path)
    if not path.exists(save_path):
        makedirs(save_path)
    if model == 'Debye': tag = 0
    else: tag = 1
    A = []
    headers = []
    for c, key in enumerate(sorted(pm.keys())):
        A.append(list(np.array(pm[key]).ravel()))
        key = model[:2]+"_"+key
        if len(A[c]) == 1:
            headers.append(key)
        else:
            for i in range(len(A[c])):
                headers.append(key+"%d" %(i+tag))
    headers = ','.join(headers)
    results = np.array(flatten(A))
    if model in ["DDebye", "PDecomp"]:
        tau_ = sol["data"]["tau"]
        add = ["tau"+"%d"%(i+1) for i in range(len(tau_))]
        add = ',' + ','.join(add)
        headers += add
        results = np.concatenate((results,tau_))
    np.savetxt(save_path+'INV_%s_%s.csv' %(model,sample_name), results[None],
               header=headers, comments='', delimiter=',')
    vars_ = ["%s"%x for x in MDL.stochastics]+["%s"%x for x in MDL.deterministics]
    if "zmod" in vars_: vars_.remove("zmod")
    MDL.write_csv(save_path+'STATS_%s_%s.csv' %(model,sample_name), variables=(vars_))

def merge_results(sol,files):
    model = get_model_type(sol)
    save_where = '/Batch results/'
    actual_path = str(path.dirname(path.realpath(argv[0]))).replace("\\", "/")
    save_path = actual_path+save_where
    print("\nMerging csv files")
    if not path.exists(save_path):
        makedirs(save_path)
    to_merge = actual_path+"/Results/%s/INV_%s_%s.csv" %(files[0],model,files[0])
    headers = np.genfromtxt(to_merge, delimiter=",", dtype=str, skip_footer=1)
    merged_inv_results = np.empty((len(files), len(headers)))
    for i, f in enumerate(files):
        merged_inv_results[i] = np.loadtxt(actual_path+"/Results/%s/INV_%s_%s.csv" %(f,model,f), delimiter=",", skiprows=1)
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

def plot_deviance(sol, save=False, draw=True, save_as_png=True):
    if save_as_png:
        save_as = 'png'
    else:
        save_as = 'pdf'
    filename = sol["path"].replace("\\", "/").split("/")[-1].split(".")[0]
    model = get_model_type(sol)
    if draw or save:
        fig, ax = plt.subplots(figsize=(6,4))
        deviance = sol["pymc_model"].trace('deviance')[:]
        sampler_state = sol["pymc_model"].get_state()["sampler"]
        x = np.arange(sampler_state["_burn"]+1, sampler_state["_iter"]+1, sampler_state["_thin"])
        plt.plot(x, deviance, "-b", linewidth=2, label="Model deviance\nDIC = %.2f\nBPIC = %.2f" %(sol["pymc_model"].DIC,sol["pymc_model"].BPIC))
        plt.xlabel("Iteration", fontsize=14)
        plt.ylabel("Deviance", fontsize=14)
        plt.yticks(fontsize=14), plt.xticks(fontsize=14)
        plt.legend(numpoints=1, fontsize=14, loc="best")
        fig.tight_layout()
    if save:
        save_where = '/Figures/ModelDeviance/'
        actual_path = str(path.dirname(path.realpath(argv[0]))).replace("\\", "/")
        save_path = actual_path+save_where
        print("\nSaving model deviance figure in:\n", save_path)
        if not path.exists(save_path):
            makedirs(save_path)
        fig.savefig(save_path+'ModelDeviance-%s-%s.%s'%(model,filename,save_as))
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

def plot_logp(sol, save=False, draw=True, save_as_png=True):
    if save_as_png:
        save_as = 'png'
    else:
        save_as = 'pdf'
    filename = sol["path"].replace("\\", "/").split("/")[-1].split(".")[0]
    model = get_model_type(sol)
    if draw or save:
        fig, ax = plt.subplots(figsize=(6,4))
        logp = logp_trace(sol["pymc_model"])
        sampler_state = sol["pymc_model"].get_state()["sampler"]
        x = np.arange(sampler_state["_burn"]+1, sampler_state["_iter"]+1, sampler_state["_thin"])
        plt.plot(x, logp, "-b", linewidth=2, label="logp")
        plt.xlabel("Iteration", fontsize=14)
        plt.ylabel("Log-likelihood", fontsize=14)
        plt.yticks(fontsize=14), plt.xticks(fontsize=14)
        plt.legend(numpoints=1, fontsize=14, loc="best")
        fig.tight_layout()
    if save:
        save_where = '/Figures/LogLikelihood/'
        actual_path = str(path.dirname(path.realpath(argv[0]))).replace("\\", "/")
        save_path = actual_path+save_where
        print("\nSaving logp trace figure in:\n", save_path)
        if not path.exists(save_path):
            makedirs(save_path)
        fig.savefig(save_path+'LogLikelihood-%s-%s.%s'%(model,filename,save_as))
    try:    plt.close(fig)
    except: pass
    if draw:    return fig
    else:       return None

def plot_fit(sol, save=False, draw=True, save_as_png=True):
    if save_as_png:
        save_as = 'png'
    else:
        save_as = 'pdf'
    filepath = sol["path"]
    sample_name = filepath.replace("\\", "/").split("/")[-1].split(".")[0]
    model = get_model_type(sol)
    data = sol["data"]
    fit = sol["fit"]
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
        plt.errorbar(zn_fit.real, -zn_dat.imag, zn_err.imag, zn_err.real, '.', label='Data')
        plt.plot(zn_fit.real, -zn_fit.imag, '-', label='Fitted model')
        plt.fill_between(zn_fit.real, -zn_max.imag, -zn_min.imag, color='#7f7f7f', alpha=0.2)
        plt.xlabel(sym_labels['real'])
        plt.ylabel(sym_labels['imag'])
        plt.legend(loc='best', fontsize=9, numpoints=1)
        plt.xlim([None, 1])
        plt.ylim([0, max(-zn_dat.imag)])
        
        # Freq-Ampl
        plt.axes(ax[1])
        plt.errorbar(f, Amp_dat, Amp_err, None, '.', label='Data')
        plt.semilogx(f, Amp_fit, '-', label='Fitted model')
        plt.fill_between(f, Amp_max, Amp_min, color='#7f7f7f', alpha=0.2)
        plt.xlabel(sym_labels['freq'])
        plt.ylabel(sym_labels['ampl'])
        plt.legend(loc='best', fontsize=9, numpoints=1)
        plt.xlim([10**np.floor(min(np.log10(f))), 10**np.ceil(max(np.log10(f)))])
        plt.ylim([None,1.0])

        # Freq-Phas
        plt.axes(ax[0])
        plt.errorbar(f, -Pha_dat, Pha_err, None, '.', label='Data')
        plt.loglog(f, -Pha_fit, '-', label='Fitted model')
        ax[0].set_yscale("log", nonposy='clip')
        plt.fill_between(f, -Pha_max, -Pha_min, color='#7f7f7f', alpha=0.2)
        plt.xlabel(sym_labels['freq'])
        plt.ylabel(sym_labels['phas'])
        plt.legend(loc='best', fontsize=9, numpoints=1)
        plt.xlim([10**np.floor(min(np.log10(f))), 10**np.ceil(max(np.log10(f)))])
        plt.ylim([1,10**np.ceil(max(np.log10(-Pha_dat)))])

        if  (-Pha_dat < 1).any() and (-Pha_dat >= 0.1).any():
            plt.ylim([0.1,10**np.ceil(max(np.log10(-Pha_dat)))])  
        if  (-Pha_dat < 0.1).any() and (-Pha_dat >= 0.01).any():
            plt.ylim([0.01,10**np.ceil(max(np.log10(-Pha_dat)))]) 
            
        plt.tight_layout(pad=1, h_pad=0, w_pad=0)
        
    if save:
        save_where = '/Figures/Fit figures/'
        actual_path = str(path.dirname(path.realpath(argv[0]))).replace("\\", "/")
        save_path = actual_path+save_where
        print("\nSaving fit figure in:\n", save_path)
        if not path.exists(save_path):
            makedirs(save_path)
        fig.savefig(save_path+'FIT-%s-%s.%s'%(model,sample_name,save_as), bbox_inches='tight')
    try:    plt.close(fig)
    except: pass
    if draw:    return fig
    else:       return None

def print_diagn(M, q, r, s):
    return raftery_lewis(M, q, r, s, verbose=0)

def plot_par():
    from matplotlib.pyplot import rcParams
#    rc = rcParams.items()    
    rc = {u'figure.dpi': 72.0,
          u'figure.edgecolor': 'white',
          u'figure.facecolor': 'white',
          u'figure.figsize': [1.0, 1.0],
          u'figure.frameon': True,
          u'figure.max_open_warning': 20,
          u'figure.subplot.bottom': 0.125,
          u'figure.subplot.hspace': 0.2,
          u'figure.subplot.left': 0.125,
          u'figure.subplot.right': 0.9,
          u'figure.subplot.top': 0.88,
          u'figure.subplot.wspace': 0.2,
          u'font.size': 10.0,
          u'font.stretch': u'normal',
          u'font.style': u'normal',
          u'font.variant': u'normal',
          u'font.weight': u'medium',
          u'grid.alpha': 1.0,
          u'grid.color': u'#b0b0b0',
          u'grid.linestyle': u'-',
          u'grid.linewidth': 0.8,
          u'legend.fontsize': 10,
          u'mathtext.bf': u'serif:bold',
          u'mathtext.cal': u'cursive',
          u'mathtext.default': u'regular',
          u'mathtext.fallback_to_cm': True,
          u'mathtext.fontset': u'stixsans',
          u'mathtext.it': u'sans:italic',
          u'mathtext.rm': u'serif',
          u'mathtext.sf': u'sans',
          u'mathtext.tt': u'monospace',
          u'savefig.bbox': u'tight',
          u'savefig.directory': u'~',
          u'savefig.dpi': 200.0,
          u'savefig.edgecolor': u'white',
          u'savefig.facecolor': u'white',
          }
    return rc
