#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 10:07:10 2017

@author: Charles
"""

import numpy as np
from past.builtins import basestring
import os

# =============================================================================
def save_figure(fig, subfolder, fname='Untitled', dpi=144):
    """
    Called at the end of any plot function attribute of 
    mcmcinv object if save=True is passed to the plot function
    """
    folder = 'Figures'
    cwd = os.getcwd().replace("\\", "/")
    save_path = cwd+"/"+folder+"/"+subfolder+"/"
    print("\nSaving figure:\n", save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    fig.savefig(save_path+fname, dpi=dpi, bbox_inches='tight')

# =============================================================================
def split_filepath(p):
    """
    Splits filepath into the name of the sample
    Pass string returns string
    """
    return p.replace("\\", "/").split("/")[-1].split(".")[0]

#==============================================================================
# To extract important information from the model (MDL)
# Used at the end of inversion routine
# Arguments: model <pymc model object>, maximum amplitude measured <float>
def format_results(M, Z_max):
    var_keys = [s.__name__ for s in M.stochastics] + [d.__name__ for d in M.deterministics]
    var_keys = [s for s in var_keys if s not in ["zmod", "mp", "cond", ]]
    Mst = M.stats(chain=-1)
    pm = {k: Mst[k]["mean"] for k in var_keys}
    pm.update({k+"_std": Mst[k]["standard deviation"] for k in var_keys})
    pm.update({"R0": Z_max*pm["R0"],"R0_std": Z_max*pm["R0_std"]}) # remove normalization
    pm.update({k.replace("log_", ""): 10**pm[k] for k in var_keys if k.startswith("log_")})
    pm.update({(k.replace("log_", ""))+"_std": abs(pm[k+"_std"]/pm[k])*(10**pm[k]) for k in var_keys if k.startswith("log_")})
    pm = {k: v for (k, v) in list(pm.items()) if "log_" not in k}
    return pm           # returns parameters and uncertainty
    
#==============================================================================
def get_data(filename,headers,ph_units):
    """
    To import data
    Arguments: file name, number of header lines to skip, phase units
    """
    # Importation des données .DAT
    dat_file = np.loadtxt("%s"%(filename),skiprows=headers,delimiter=',')
    labels = ["freq", "amp", "pha", "amp_err", "pha_err"]
    data = {l:dat_file[:,i] for (i,l) in enumerate(labels)}
    if ph_units == "mrad":
        data["pha"] = data["pha"]/1000                    # mrad to rad
        data["pha_err"] = data["pha_err"]/1000              # mrad to rad
    if ph_units == "deg":
        data["pha"] = np.radians(data["pha"])               # deg to rad
        data["pha_err"] = np.radians(data["pha_err"])       # deg to rad
    data["phase_range"] = abs(max(data["pha"])-min(data["pha"])) # Range of phase measurements (used in NRMS error calculation)
    data["Z"]  = data["amp"]*(np.cos(data["pha"]) + 1j*np.sin(data["pha"]))
    EI = np.sqrt(((data["amp"]*np.cos(data["pha"])*data["pha_err"])**2)+(np.sin(data["pha"])*data["amp_err"])**2)
    ER = np.sqrt(((data["amp"]*np.sin(data["pha"])*data["pha_err"])**2)+(np.cos(data["pha"])*data["amp_err"])**2)
    data["Z_err"] = ER + 1j*EI
    # Normalization of amplitude
    data["Z_max"] = max(abs(data["Z"]))  # Maximum amplitude
    zn, zn_e = data["Z"]/data["Z_max"], data["Z_err"]/data["Z_max"] # Normalization of impedance by max amplitude
    data["zn"] = np.array([zn.real, zn.imag]) # 2D array with first column = real values, second column = imag values
    data["zn_err"] = np.array([zn_e.real, zn_e.imag]) # 2D array with first column = real values, second column = imag values
    return data

# =============================================================================
def get_model_type(sol):
    """
    Pass mcmcinv object
    Returns a string with the specifics about 
    the model type e.g. c_exp in Debye decomposition or 
    n_modes in ColeCole model
    """
    model = sol.model
    if model in ["PDecomp", "CCD"]:
        if sol.model_type["c_exp"] == 0.5:
            model = "WarburgDecomp"
        elif sol.model_type["c_exp"] == 1.0:
            model = "DebyeDecomp"
        else:
            model = "ColeColeDecomp"
        
        model = ''.join([c for c in model if c.isupper()])
    if model == "ColeCole":
        model = "CC%d"%sol.cc_modes
    return model

# =============================================================================
def var_depth(var):
    """
    Pass stochastic or deterministic
    returns the size of the extra dimensions
    (other than the one with length nb_iter-nb_burn)
    """
    return int(var.trace().size/len(var.trace()))

# =============================================================================  
def flatten(x):
    """
    Flattens a N-D list
    """
    result = []
    for el in x:
        if hasattr(el, "__iter__") and not isinstance(el, basestring):
            result.extend(flatten(el))
        else:
            result.append(el)
    return result

# =============================================================================
def find_nearest(array,val):
    """
    Returns the nearest value to val in an array
    """
    idx = (np.abs(array-val)).argmin()
    return array[idx]
