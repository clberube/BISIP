#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 10:07:10 2017

@author: Charles
"""

import numpy as np

#==============================================================================
# To extract important information from the model (MDL)
# Used at the end of inversion routine
# Arguments: model <pymc model object>, maximum amplitude measured <float>
def format_results(M, Z_max):
    var_keys = [s.__name__ for s in M.stochastics] + [d.__name__ for d in M.deterministics]
    var_keys = [s for s in var_keys if s not in ["zmod", "mp"]]
    Mst = M.stats(chain=-1)
    pm = {k: Mst[k]["mean"] for k in var_keys}
    pm.update({k+"_std": Mst[k]["standard deviation"] for k in var_keys})
    pm.update({"R0": Z_max*pm["R0"],"R0_std": Z_max*pm["R0_std"]}) # remove normalization
    pm.update({k.replace("log_", ""): 10**pm[k] for k in var_keys if k.startswith("log_")})
    pm.update({(k.replace("log_", ""))+"_std": abs(pm[k+"_std"]/pm[k])*(10**pm[k]) for k in var_keys if k.startswith("log_")})
    pm = {k: v for (k, v) in list(pm.items()) if "log_" not in k}
    return pm           # returns parameters and uncertainty
    
#==============================================================================
# To import data
# Arguments: file name, number of header lines to skip, phase units
def get_data(filename,headers,ph_units):
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