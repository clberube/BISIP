#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: clberube
# @Date:   05-03-2020
# @Email:  charles.lafreniere-berube@polymtl.ca
# @Last modified by:   charles
# @Last modified time: 2020-03-23T15:49:25-04:00


import numpy as np
import matplotlib.pyplot as plt
from corner import corner


class plotlib(object):

    def plot_traces(self, chain=None, **kwargs):
        """
        Plots the traces of the MCMC simulation.

        Args:
            chain (:obj:`ndarray`): A numpy array containing the MCMC chain to
                plot. Should have a shape (nsteps, nwalkers, ndim) or
                (nsteps*nwalkers, ndim). If None, the full, unflattened chain
                will be used and all walkers will be plotted. Defaults to None.
            **kwargs: Additional keyword arguments for the get_chain function
                (see below). Use these arguments only if not explicitly passing
                the `chain` argument.

        Keyword Args:
            discard (:obj:`int`): The number of steps to discard.
            thin (:obj:`int`): The thinning factor (keep every `thin` step).
            flat (:obj:`bool`): Whether to flatten the walkers into a single
                chain or not.

        Returns:
            :obj:`Figure`: A matplotlib figure.

        """
        self._check_if_fitted()
        if chain is None:
            chain = self.get_chain(**kwargs)
        labels = self.param_names
        fig, axes = plt.subplots(self.ndim, figsize=(8, 6), sharex=True)
        for i in range(self.ndim):
            ax = axes[i]
            ax.plot(chain[:, :, i], 'k', alpha=0.3)
            ax.set_xlim(0, len(chain))
            ax.set_ylim(self.param_bounds[:, i])
            ax.set_ylabel(labels[i])
            ax.yaxis.set_label_coords(-0.1, 0.5)
        axes[-1].set_xlabel('Steps')
        fig.tight_layout()
        return fig

    def plot_histograms(self, chain=None, bins=25, **kwargs):
        """
        Plots histograms of the MCMC simulation chains.

        Args:
            chain (:obj:`ndarray`): A numpy array containing the MCMC chain to
                plot. Should have a shape (nsteps, nwalkers, ndim) or
                (nsteps*nwalkers, ndim). If None, the full, unflattened chain
                will be used and all walkers will be plotted. Defaults to None.
            bins (:obj:`int`): The number of bins to use in the histograms.
            **kwargs: Additional keyword arguments for the get_chain function
                (see below). Use these arguments only if not explicitly passing
                the `chain` argument.

        Keyword Args:
            discard (:obj:`int`): The number of steps to discard.
            thin (:obj:`int`): The thinning factor (keep every `thin` step).
            flat (:obj:`bool`): Whether to flatten the walkers into a single
                chain or not.

        Returns:
            :obj:`Figure`: A matplotlib figure.

        """
        self._check_if_fitted()
        chain = self.parse_chain(chain, **kwargs)
        labels = self.param_names
        fig, axes = plt.subplots(self.ndim, figsize=(5, 1.5*chain.shape[1]))
        for i in range(self.ndim):
            ax = axes[i]
            ax.hist(chain[:, i], bins=bins, fc='w', ec='k')
            ax.set_xlabel(labels[i])
            ax.ticklabel_format(axis='x', scilimits=[-2, 2])
        fig.tight_layout()
        return fig

    def plot_fit(self, chain=None, p=[2.5, 50, 97.5], **kwargs):
        """
        Plots the input data, best fit and confidence interval of a model.
        Shows the real and imaginary parts.

        Args:
            chain (:obj:`ndarray`): A numpy array containing the MCMC chain to
                plot. Should have a shape (nsteps, nwalkers, ndim) or
                (nsteps*nwalkers, ndim). If None, the full, unflattened chain
                will be used and all walkers will be plotted. Defaults to None.
            p (:obj:`list` of :obj:`int`): Percentile values for lower
                confidence interval, best fit curve, and upper confidence
                interval, **in that order**. Defaults to [2.5, 50, 97.5] for
                the median and 95% HPD.
            **kwargs: Additional keyword arguments for the get_chain function
                (see below). Use these arguments only if not explicitly passing
                the `chain` argument.

        Keyword Args:
            discard (:obj:`int`): The number of steps to discard.
            thin (:obj:`int`): The thinning factor (keep every `thin` step).
            flat (:obj:`bool`): Whether to flatten the walkers into a single
                chain or not.

        Returns:
            :obj:`Figure`: A matplotlib figure.

        """
        self._check_if_fitted()
        data = self.data
        lines = self.get_model_percentile(p, chain, **kwargs)
        fig, ax = plt.subplots(1, 2, figsize=(8, 3))
        for i in range(2):
            ax[i].errorbar(data['freq'], data['zn'][i], yerr=data['zn_err'][i],
                           markersize=3, fmt=".k", capsize=0)
            ax[i].plot(data['freq'], lines[0][i], ls=':', c='0.5')
            ax[i].plot(data['freq'], lines[1][i], c='C3')
            ax[i].plot(data['freq'], lines[2][i], ls=':', c='0.5')
            ax[i].set_ylabel(r'$\rho${} (normalized)'.format((i+1)*"'"))
            # ax[i].yaxis.set_label_coords(-0.2, 0.5)
            ax[i].set_xscale('log')
            ax[i].set_xlabel('$f$ (Hz)')
        fig.tight_layout()
        return fig

    def plot_data(self, feature='phase', **kwargs):
        """
        Plots the input data.

        Args:
            feature (:obj:`str`): Which data feature to plot. Choices are
                'phase', 'amplitude', 'real', 'imaginary'. Defaults to 'phase'.
            **kwargs: Additional keyword arguments passed to the matplotlib
                errorbar function.

        Returns:
            :obj:`Figure`: A matplotlib figure.

        """
        kwargs.setdefault('fmt', '.k')
        kwargs.setdefault('capsize', 0)
        kwargs.setdefault('markersize', 3)

        data = self.data
        fig, ax = plt.subplots()
        x = data['freq']
        y = {'phase': (-data['pha'],
                       data['pha_err'],
                       '-Phase (rad)'),
             'amplitude': (data['amp']/data['norm_factor'],
                           data['amp_err']/data['norm_factor'],
                           'Amplitude (normalized)'),
             'real': (data['Z'][0]/data['norm_factor'],
                      data['Z_err'][0]/data['norm_factor'],
                      'Real part (normalized)'),
             'imaginary': (-data['Z'][1]/data['norm_factor'],
                           data['Z_err'][1]/data['norm_factor'],
                           '-Imaginary part (normalized)'),
             }
        ax.errorbar(x, y[feature][0], yerr=y[feature][1], **kwargs)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel(y[feature][2])
        ax.set_xscale('log')
        fig.tight_layout()
        return fig

    def plot_fit_pa(self, chain=None, p=[2.5, 50, 97.5], **kwargs):
        """
        Plots the input data, best fit and confidence interval of a model.
        Shows the amplitude and phase spectra.

        Args:
            chain (:obj:`ndarray`): A numpy array containing the MCMC chain to
                plot. Should have a shape (nsteps, nwalkers, ndim) or
                (nsteps*nwalkers, ndim). If None, the full, unflattened chain
                will be used and all walkers will be plotted. Defaults to None.
            p (:obj:`list` of :obj:`int`): Percentile values for lower
                confidence interval, best fit curve, and upper confidence
                interval, **in that order**. Defaults to [2.5, 50, 97.5] for
                the median and 95% HPD.
            **kwargs: Additional keyword arguments for the get_chain function
                (see below). Use these arguments only if not explicitly passing
                the `chain` argument.

        Keyword Args:
            discard (:obj:`int`): The number of steps to discard.
            thin (:obj:`int`): The thinning factor (keep every `thin` step).
            flat (:obj:`bool`): Whether to flatten the walkers into a single
                chain or not.

        Returns:
            :obj:`Figure`: A matplotlib figure.

        """
        self._check_if_fitted()
        data = self.data
        lines = self.get_model_percentile(p, chain, **kwargs)
        fig, ax = plt.subplots(1, 2, figsize=(8, 3))

        ax[0].errorbar(data['freq'], data['amp']/data["norm_factor"],
                       yerr=data['amp_err']/data["norm_factor"],
                       markersize=3, fmt=".k", capsize=0)
        ax[0].plot(data['freq'], np.linalg.norm(lines[0], axis=0), ls=':', c='0.5')
        ax[0].plot(data['freq'], np.linalg.norm(lines[1], axis=0), c='C3')
        ax[0].plot(data['freq'], np.linalg.norm(lines[2], axis=0), ls=':', c='0.5')
        ax[0].set_ylabel('Amplitude (normalized)')

        ax[1].errorbar(data['freq'], -data['pha'], yerr=data['pha_err'],
                       markersize=3, fmt=".k", capsize=0)
        ax[1].plot(data['freq'], -np.arctan2(*lines[0][::-1]), ls=':', c='0.5')
        ax[1].plot(data['freq'], -np.arctan2(*lines[1][::-1]), c='C3')
        ax[1].plot(data['freq'], -np.arctan2(*lines[2][::-1]), ls=':', c='0.5')
        ax[1].set_ylabel('-Phase (rad)')
        ax[1].set_yscale('log')

        for i in range(2):
            ax[i].set_xscale('log')
            ax[i].set_xlabel('$f$ (Hz)')
        fig.tight_layout()
        return fig

    def plot_corner(self, chain=None, **kwargs):
        """
        Plots the corner plot of the MCMC simulation.

        Args:
            chain (:obj:`ndarray`): A numpy array containing the MCMC chain to
                plot. Should have a shape (nsteps, nwalkers, ndim) or
                (nsteps*nwalkers, ndim). If None, the full, unflattened chain
                will be used and all walkers will be plotted. Defaults to None.
            **kwargs: Additional keyword arguments for the get_chain function
                (see below). Use these arguments only if not explicitly passing
                the `chain` argument.

        Keyword Args:
            discard (:obj:`int`): The number of steps to discard.
            thin (:obj:`int`): The thinning factor (keep every `thin` step).
            flat (:obj:`bool`): Whether to flatten the walkers into a single
                chain or not.

        Returns:
            :obj:`Figure`: A matplotlib figure.

        """
        self._check_if_fitted()
        chain = self.parse_chain(chain, **kwargs)
        fig = corner(chain, labels=self.param_names)
        return fig
