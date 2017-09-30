# -*- coding: utf-8 -*-

"""
Created on Tue Apr 21 12:05:22 2015

@author:    charleslberube@gmail.com
            École Polytechnique de Montréal

The MIT License (MIT)

Copyright (c) 2015-2016 Charles L. Bérubé

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

This python script builds the graphical user interface may be used to call the
Bayesian inversion module of all SIP models (BISIP_models.py)
"""

from __future__ import division
from __future__ import print_function
#from __future__ import unicode_literals

#==============================================================================
# System imports

from sys import version_info
print("Running Python %d.%d.%d"%version_info[0:3])

print("\nFuture imports")
from future import standard_library
standard_library.install_aliases()
from builtins import zip
from builtins import str
from builtins import map
from builtins import range
from past.builtins import basestring
from builtins import object
from past.utils import old_div

print("System imports")
from sys import argv, stdout
stdout.flush()
from platform import system
from os.path import realpath as osp_realpath
from json import load as jload, dump as jdump
from warnings import filterwarnings
filterwarnings('ignore') # Ignore some tkinter warnings

print("GUI imports")
if version_info[0] < 3:
    import Tkinter as tk
    import FixTk # To avoid pyinstaller error
else:
    import tkinter as tk
import tkinter.filedialog, tkinter.messagebox, tkinter.font

print("BISIP imports")
from bisip.models import mcmcinv
import bisip.invResults as iR

print("Other imports")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.pyplot import rcParams
from itertools import combinations
from collections import OrderedDict

print("All modules successfully loaded")
stdout.flush()

#==============================================================================
# Fonts
if "Darwin" in system():
    print("\nOS X detected")
    fontsize = 12
    pad_radio = 3
    but_size = -2
    res_size = -1
else:
    print("\nWindows detected")
    fontsize = 10
    pad_radio = 0
    but_size = -2
    res_size = -2

window_font = "TkDefaultFont %s"%fontsize
fontz = {"bold": ("TkDefaultFont", fontsize, "bold"),
         "normal_small": ("TkDefaultFont", fontsize+but_size, "normal"),
         "italic_small": ("TkDefaultFont", fontsize+but_size, "italic")}

# To flatten ND lists
def flatten(x):
    result = []
    for el in x:
        if hasattr(el, "__iter__") and not isinstance(el, basestring):
            result.extend(flatten(el))
        else:
            result.append(el)
    return result

#==============================================================================
# Import last used window parameters
# All GUI options and choices saved when closing the main window
# Parameters are saved in root_ini file in the executable's directory
# To reset and use default parameters, delete root_ini in local directory

class MainApplication(object):
    working_path = str(osp_realpath(argv[0])).replace("\\", "/")+"/"
    def __init__(self, master, fontz):
        self.save_options = {"Save all hexbins (will make error)":            tk.BooleanVar(),
                             "Save all bivariate KDE (will make error)":      tk.BooleanVar(),
                             "Save fit figures":            tk.BooleanVar(),
                             "Save traces":                 tk.BooleanVar(),
                             "Save histograms":             tk.BooleanVar(),
                             "Save autocorrelations":       tk.BooleanVar(),
                             "Save Debye RTD":              tk.BooleanVar(),
                             "Save summaries":              tk.BooleanVar(),
                             "Save deviance":               tk.BooleanVar(),
                             "Save loglikelihood":          tk.BooleanVar(),
                             "Save traces as txt":          tk.BooleanVar(),
                             "Tuning verbose":              tk.BooleanVar(),
                             "No subplots":                 tk.BooleanVar(),
                             "PNG figures":                 tk.BooleanVar(),
                            }
        self.run_options = OrderedDict((
                            ("Auto draw fit",                tk.BooleanVar()),
                            ("Print results in console",     tk.BooleanVar()),
                            ("Save CSV results",             tk.BooleanVar()),
                            ))

        self.mcmc_vars = OrderedDict((
                                ("Number of chains"     , (tk.IntVar(), 1)),
                                ("Total iterations"     , (tk.IntVar(), 100000)),
                                ("Burn-in period"       , (tk.IntVar(), 80000)),
                                ("Thinning factor"      , (tk.IntVar(), 1)),
                                ("Tuning interval"      , (tk.IntVar(), 10000)),
                                ("Proposal scale"       , (tk.DoubleVar(), 1)),
                                ("Covariance delay"     , (tk.IntVar(), 10000)),
                                ("Covariance interval"  , (tk.IntVar(), 10000)),
                                ))

        self.master = master
        self.master.resizable(width=tk.FALSE, height=tk.FALSE)
        self.load()
        self.set_plot_par()
        self.build_helpmenu()
        self.make_main_frames()
        self.make_browse_button()
        self.headers_input()
        self.open_files = self.root_ini["Imported files"]
        self.draw_file_list()
        self.phase_units()
        self.mcmc_parameters()
        self.run_exit()
        self.make_options(from_root=True)
        self.model_choice()
        self.activity(idle=True)

    def use_default_root_ini(self):
        self.default_root = {
                            "Spectral IP model"     : "ColeCole",
                            "Adaptive Metropolis"   : True,
                            "Nb header lines"       : 1,
                            "Phase units"           : "mrad",
                            "Imported files"        : [],
                            "Polyn order"           : 4,
                            "Freq dep"              : 1.0,
                            "Nb modes"              : 2,
                            }
        for k, v in list(self.save_options.items()):
            self.default_root[k] = v.get()
        for k, v in list(self.mcmc_vars.items()):
            self.default_root[k] = v[1]
        for k, v in list(self.run_options.items()):
            self.default_root[k] = v.get()
        return self.default_root

#==============================================================================
# Main frames
#==============================================================================
    def make_main_frames(self):
        # Frame for importing files
        self.frame_import = tk.LabelFrame(self.master, text="1. Import data", width=200, height=5, font=fontz["bold"])
        self.frame_import.grid(row = 0, column=1, columnspan=1, rowspan=5, sticky=tk.W+tk.E+tk.N, padx=10, pady=(5,15))
        self.frame_import.grid_columnconfigure(0, weight=1), self.frame_import.grid_rowconfigure(0, weight=1)
        # Frame for choosing the model
        self.frame_model = tk.LabelFrame(self.master, text="2. SIP model", width=200, height=4, font=fontz["bold"])
        self.frame_model.grid(row = 6, column=1, columnspan=1, sticky=tk.W+tk.E+tk.N, padx=10, pady=(5,15))
        self.frame_model.grid_columnconfigure(0, weight=1), self.frame_model.grid_rowconfigure(0, weight=1)
        # Frame to enter mcmc parameters
        self.frame_mcmc = tk.LabelFrame(self.master, text="3. MCMC settings", width=200, height=4, font=fontz["bold"])
        self.frame_mcmc.grid(row = 7, column=1, columnspan=1, sticky=tk.W+tk.E+tk.N, padx=10, pady=(5,15), ipady=3)
        self.frame_mcmc.grid_columnconfigure(0, weight=1)
        # Frame to run and exit
        self.frame_ruex = tk.LabelFrame(self.master, text="4. Run", width=200, height=4, font=fontz["bold"])
        self.frame_ruex.grid(row = 8, column=1, columnspan=1, sticky=tk.S+tk.W+tk.E+tk.N, padx=10, pady=(5,10))
        self.frame_ruex.columnconfigure(0, weight=1)
        # Frame to list the imported files and preview
        self.frame_list = tk.LabelFrame(self.master, text="List of imported files", font=fontz["bold"])
        self.frame_list.grid(row = 0, column=0, columnspan=1, rowspan=15, sticky=tk.W+tk.E+tk.N+tk.S, padx=10, pady=(5,10))
        self.frame_list.grid_rowconfigure(1, weight=1), self.frame_list.columnconfigure(0, weight=1)
        # Frame to visualize results
        self.frame_results = tk.LabelFrame(self.master, text="Results", font=fontz["bold"])
        self.frame_results.grid(row = 0, column=2, columnspan=1, rowspan=15, sticky=tk.W+tk.E+tk.N+tk.S, padx=10, pady=(5,10))
        self.frame_results.grid_rowconfigure(14, weight=1)

#==============================================================================
# Running inversion
#==============================================================================
    def run_inversion(self):
        try:    self.clear()
        except: pass
        self.sel_files = [str(self.open_files[i]) for i in self.text_files.curselection()]
        if len(self.sel_files) == 0:
            tkinter.messagebox.showwarning("Inversion error",
                                     "No data selected for inversion \nSelect at least one data file in the left panel", parent=self.master)
        if len(self.sel_files) >= 1:
            try:
                self.Inversion()
                stdout.flush()
            except:
                tkinter.messagebox.showerror("Inversion error", "Error\nMake sure all fields are OK\nMake sure data file is correctly formatted",
                                             parent=self.master)
        return

    def Inversion(self):
        print("\n")
        print("=====================")
        print("Starting inversion...")
        print("=====================")

        print("Model:", self.model.get())
        if self.model.get() == "ColeCole":
            print("Cole-Cole modes:", self.modes_n.get())
        if self.model.get() == "PDecomp":
            print("Polynomial order:", self.poly_n.get())
            if self.c_exp.get() == 1.0:
                decomp_type = "(Debye)"
            elif self.c_exp.get() == 0.5:
                decomp_type = "(Warburg)"
            else:
                decomp_type = "(Cole-Cole)"
            print("Frequency dependence:", self.c_exp.get(), decomp_type)
        print("Units:", self.units.get())
        print("Paths:")
        for i in self.sel_files:
            print(i)
        print("Skipping", self.head.get(), "header lines")
        self.files = [self.sel_files[i].split("/")[-1].split(".")[0] for i in range(len((self.sel_files)))]

        self.mcmc_params = {"adaptive"   : self.adaptive.get(),
                            "verbose"    : self.save_options["Tuning verbose"].get(),
                            "nb_chain"   : self.mcmc_vars["Number of chains"][0].get(),
                            "nb_iter"    : self.mcmc_vars["Total iterations"][0].get(),
                            "nb_burn"    : self.mcmc_vars["Burn-in period"][0].get(),
                            "thin"       : self.mcmc_vars["Thinning factor"][0].get(),
                            "tune_inter" : self.mcmc_vars["Tuning interval"][0].get(),
                            "prop_scale" : self.mcmc_vars["Proposal scale"][0].get(),
                            "cov_inter"  : self.mcmc_vars["Covariance interval"][0].get(),
                            "cov_delay"  : self.mcmc_vars["Covariance delay"][0].get(),
                            }
        # Appel de la fonction d'inversion avec les paramètres sélectionnés
        try:    del(self.all_results)
        except: pass
        self.all_results = {}
        self.draw_drop_down()
        self.var_review.set("Working...")
        for (i, self.f_n) in enumerate(self.files):
            print("=====================")
            self.activity()
            self.var_review.set(self.f_n)
            self.sol = mcmcinv(   self.model.get(), self.sel_files[i], mcmc = self.mcmc_params,
                                headers=self.head.get(), ph_units=self.units.get(),
                                cc_modes=self.modes_n.get(), decomp_poly=self.poly_n.get(),
                                c_exp=self.c_exp.get(), keep_traces=self.save_options["Save traces as txt"].get())

            self.all_results[self.f_n] = {"pm":self.sol["params"],"MDL":self.sol["pymc_model"],"data":self.sol["data"],"fit":self.sol["fit"], "sol":self.sol}
#           Impression ou non des résultats, graphiques, histogrammes
            try:            
                self.update_results()        
            except:
                print("PROBLEM")
            if self.run_options["Print results in console"].get():
                iR.print_resul(self.sol)
            if self.run_options["Save CSV results"].get():
                iR.save_resul(self.sol)
            fig_fit = iR.plot_fit(self.sol, save=self.save_options["Save fit figures"].get(), save_as_png=self.save_options["PNG figures"].get(), draw=self.run_options["Auto draw fit"].get())
            if self.run_options["Auto draw fit"].get():
                self.plot_window(fig_fit, "Inversion results: "+self.f_n)
            if self.model.get() == "PDecomp":
                iR.plot_debye(self.sol, save=self.save_options["Save fit figures"].get(), save_as_png=self.save_options["PNG figures"].get())
            if self.save_options["Save all hexbins (will make error)"].get():
                for v1, v2 in list(combinations(self.list_of_parameters, 2)):
                    iR.plot_hexbin(self.all_results[self.f_n]["sol"], v1, v2, save=True, save_as_png=self.save_options["PNG figures"].get())
            if self.save_options["Save all bivariate KDE (will make error)"].get():
                for v1, v2 in list(combinations(self.list_of_parameters, 2)):
                    iR.plot_KDE(self.all_results[self.f_n]["sol"], v1, v2, save=True, save_as_png=self.save_options["PNG figures"].get())
            if self.save_options["Save histograms"].get():
                try:
                    iR.plot_histo(self.all_results[self.f_n]["sol"], no_subplots=self.save_options["No subplots"].get(), save=True, save_as_png=self.save_options["PNG figures"].get())
                except:
                    pass
            if self.save_options["Save traces"].get():
                iR.plot_traces(self.all_results[self.f_n]["sol"], no_subplots=self.save_options["No subplots"].get(), save=True, save_as_png=self.save_options["PNG figures"].get())
            if self.save_options["Save summaries"].get():
                iR.plot_summary(self.all_results[self.f_n]["sol"], save=True, save_as_png=self.save_options["PNG figures"].get())
            if self.save_options["Save autocorrelations"].get():
                iR.plot_autocorr(self.all_results[self.f_n]["sol"], save=True, save_as_png=self.save_options["PNG figures"].get())
            if self.save_options["Save deviance"].get():
                iR.plot_deviance(self.all_results[self.f_n]["sol"], save=True, save_as_png=self.save_options["PNG figures"].get())
            if self.save_options["Save loglikelihood"].get():
                iR.plot_logp(self.all_results[self.f_n]["sol"], save=True, save_as_png=self.save_options["PNG figures"].get())

            
            if self.files.index(self.f_n)+1 == len(self.files):
                self.activity(done=True)
                self.diagn_buttons()
                self.write_output_path()
                print("=====================")

#==============================================================================
# Result frame
#==============================================================================
    # Batch progress
    def activity(self, idle=False, done=False):
        try:    self.frame_activ.destroy()
        except: pass
        self.frame_activ = tk.Frame(self.frame_results)
        self.frame_activ.grid(row=0, column=0, sticky=tk.E+tk.W, padx=10, pady=10)
        if idle:
            display = """ \n """
            text_act = tk.Label(self.frame_activ, text=display, anchor=tk.W, justify=tk.LEFT, width=40)
        else:
            display = """Working on:\n%s (#%d/%d)..."""%(self.f_n, self.files.index(self.f_n)+1, len(self.files))
            text_act = tk.Label(self.frame_activ, text=display, anchor=tk.W, justify=tk.LEFT, width=40)
        text_act.grid(row=0, column=0, sticky=tk.W+tk.E+tk.N)
        text_act.update()
        if done:
            text_act.destroy()
            text_done = tk.Label(self.frame_activ, text="""Done\n""", width=40, anchor=tk.W)
            text_done.grid(row=0, column=0, sticky=tk.W+tk.E+tk.N)

    def clear(self, menu=False): # self expl
        self.frame_optimal.destroy()
        self.frame_saved_in.destroy()
        self.frame_diagn_but.destroy()
        if menu:
            self.frame_drop.destroy(), self.frame_activ.destroy()

    def change_file(self):
        self.clear()
        self.update_results()
        self.diagn_buttons()
        self.write_output_path()

    def draw_drop_down(self):
        try:    self.frame_drop.destroy()
        except: pass
        self.frame_drop = tk.Frame(self.frame_results)
        self.frame_drop.grid(row=1, column=0, sticky=tk.E+tk.W, padx=10, pady=3)
        self.frame_drop.columnconfigure(0,weight=1)
        self.var_review = tk.StringVar()
        self.var_review.set(self.files[0])
        optionmenu = tk.OptionMenu(self.frame_drop, self.var_review, *self.files, command = lambda x: self.change_file())
        optionmenu.grid(row=0, column=0, sticky=tk.W+tk.E+tk.S)
        optionmenu.config(bg = "gray97", relief=tk.GROOVE)

    def merge_csv_files(self):
        if len(self.files) > 1:
            iR.merge_results(self.sol, self.files)
            print("=====================")
        else:
            print("Can't merge csv files: Only 1 file inverted in last batch")
            print("=====================")
        stdout.flush()

    def RLD_diagnostic(self):
        MDL = self.all_results[self.var_review.get()]["MDL"]
        if self.model.get() == "Debye": adj = -1
        else: adj = 0
        try:
            keys = [x.__name__ for x in MDL.stochastics]
            for (i, k) in enumerate(keys):
                vect = old_div((MDL.trace(k)[:].size),(len(MDL.trace(k)[:])))
                if vect > 1:
                 keys[i] = [k+"%d"%n for n in range(1+adj,vect+1+adj)]
            keys = list(reversed(sorted(flatten(keys))))
            top_RLD = tk.Toplevel()
            top_RLD.title("Raftery-Lewis diagnostic")
            text_RLD = tk.Text(top_RLD, width=110, font=('courier', fontsize, 'normal'))
            text_RLD.grid(stick=tk.N, padx=(20,0), pady=(10,10))
            q=0.025
            r=0.01
            s=0.95
            for k in keys:
                if k[-1] not in ["%d"%d for d in range(1+adj,8)] or k == "R0":
                    data = MDL.trace(k)[:].ravel()
                else:
                    data = MDL.trace(k[:-1])[:][:,int(k[-1])-1-adj].ravel()
                F="%s:\n"%k
                nmin, kthin, nburn, nprec, kmind = iR.print_diagn(data, q, r, s)
                A="%s iterations required (assuming independence) to achieve %s accuracy with %i percent probability.\n" %(nmin, r, 100 * s)
                B="Thinning factor of %i required to produce a first-order Markov chain.\n" %kthin
                C="%i iterations to be discarded at the beginning of the simulation (burn-in).\n" %nburn
                D="%s subsequent iterations required.\n" %nprec
                E="Thinning factor of %i required to produce an independence chain.\n\n" %kmind
                text_RLD.insert("1.0", F+A+B+C+D+E)
            text_RLD.insert("1.0", "(From PYMC)"+"\n\n")
            text_RLD.insert("1.0", self.var_review.get()+"\n\n")
            button = tk.Button(top_RLD, height=1, width=20, text="Dismiss", command=top_RLD.destroy, bg='gray97', relief=tk.GROOVE)
            button.grid(row=1, column=0, sticky=tk.S, pady=(0,10))

            s = tk.Scrollbar(top_RLD, width=20)
            s.grid(row=0, column=1, sticky=tk.E+tk.N+tk.S, padx=(0,0),pady=(10,10))
            s['command'] = text_RLD.yview
            text_RLD['yscrollcommand'] = s.set
            top_RLD.resizable(width=tk.FALSE, height=tk.FALSE)
        except:
            tkinter.messagebox.showwarning("Diagnostic error",
                                     "Error\nRun inversion first", parent=self.master)

    def popup_bivar(self, bivar_type):
        try: self.top_bivar.destroy()
        except: pass
        self.top_bivar = tk.Toplevel()
        self.top_bivar.title("Bivariate plotting")
        tk.Label(self.top_bivar, text="Select two different parameters: ").grid(row=0, column=0, sticky=tk.W+tk.E+tk.N, pady=(5,5))
        self.biv1, self.biv2 = tk.StringVar(), tk.StringVar()
        self.biv1.set(self.list_of_parameters[0])
        self.biv2.set(self.list_of_parameters[1])
        optionmenu1 = tk.OptionMenu(self.top_bivar, self.biv1, *self.list_of_parameters)
        optionmenu2 = tk.OptionMenu(self.top_bivar, self.biv2, *self.list_of_parameters)
        optionmenu1.grid(row=1, column=0, sticky=tk.W+tk.E+tk.S)
        optionmenu1.config(bg = "gray97", relief=tk.GROOVE)
        optionmenu2.grid(row=2, column=0, sticky=tk.W+tk.E+tk.S)
        optionmenu2.config(bg = "gray97", relief=tk.GROOVE)
        button = tk.Button(self.top_bivar, height=1, width=20, text="OK", command=lambda: self.plot_diagnostic(bivar_type), bg='gray97', relief=tk.GROOVE)
        button.grid(row=3, column=0, sticky=tk.S, pady=(10,10))

    # Diagnostics buttons
    def diagn_buttons(self):
        try:    self.frame_diagn_but.destroy()
        except: pass
        self.frame_diagn_but = tk.Frame(self.frame_results)
        self.frame_diagn_but.grid(row=14, column=0, sticky=tk.E+tk.W+tk.S, padx=10, pady=10)
        self.frame_diagn_but.columnconfigure((0,1,2), weight=1)
        but_cle = tk.Button(self.frame_diagn_but, height=1, width=10, text = "Clear", fg='black', bg='gray97', font=fontz["normal_small"],
                            command = lambda: self.clear(menu=True), relief=tk.GROOVE)
        but_cle.grid(row=9, column=0, columnspan=3, sticky=tk.E+tk.S, pady=(5,0))
        but_mer = tk.Button(self.frame_diagn_but, height=1, text = "Merge csv result files", fg='black', bg='gray97',
                            command = self.merge_csv_files, font=fontz["normal_small"], relief=tk.GROOVE)
        but_mer.grid(row=0, column=0, columnspan=3, sticky=tk.W+tk.E+tk.S, pady=(5,0))
        but_fit = tk.Button(self.frame_diagn_but, height=1, text = "Draw data and fit", fg='black', bg='gray97',
                            command = self.plot_fit_now, font=fontz["normal_small"], relief=tk.GROOVE)
        but_fit.grid(row=1, column=1, sticky=tk.W+tk.E+tk.S, padx=(5,0), pady=(5,0))
        but_dev = tk.Button(self.frame_diagn_but, height=1, text = "Model deviance", fg='black', bg='gray97',
                            command = lambda: self.plot_diagnostic("deviance"), font=fontz["normal_small"], relief=tk.GROOVE)
        but_dev.grid(row=1, column=0, sticky=tk.W+tk.E+tk.S, padx=(0,0), pady=(5,0))
        but_lik = tk.Button(self.frame_diagn_but, height=1, text = "Log-likelihood", fg='black', bg='gray97',
                            command = lambda: self.plot_diagnostic("logp"), font=fontz["normal_small"], relief=tk.GROOVE)
        but_lik.grid(row=1, column=2, sticky=tk.W+tk.E+tk.S, padx=(5,0), pady=(5,0))
        but_sum = tk.Button(self.frame_diagn_but, height=1, text = "Summary and Gelman-Rubin convergence", fg='black', bg='gray97',
                            command = lambda: self.plot_diagnostic("summary"), font=fontz["normal_small"], relief=tk.GROOVE)
        but_sum.grid(row=7, column=0, columnspan=3, sticky=tk.W+tk.E+tk.S, pady=(5,0))
        but_hex = tk.Button(self.frame_diagn_but, height=1, text = "Hex. binning", fg='black', bg='gray97',
                            command = lambda: self.popup_bivar("hexbin"), font=fontz["normal_small"], relief=tk.GROOVE)
        but_hex.grid(row=2, column=0, columnspan=1, sticky=tk.W+tk.E+tk.S, pady=(5,0))
        but_kde = tk.Button(self.frame_diagn_but, height=1, text = "Bivariate KDE", fg='black', bg='gray97',
                            command = lambda: self.popup_bivar("KDE"), font=fontz["normal_small"], relief=tk.GROOVE)
        but_kde.grid(row=2, column=2, columnspan=1, sticky=tk.W+tk.E+tk.S, pady=(5,0), padx=(5,0))
        but_rld = tk.Button(self.frame_diagn_but, height=1, text = "Raftery-Lewis diagnostic", fg='black', bg='gray97',
                            command = self.RLD_diagnostic, font=fontz["normal_small"], relief=tk.GROOVE)
        but_rld.grid(row=8, column=0, columnspan=3, sticky=tk.W+tk.E+tk.S, pady=(5,5))
        but_tra = tk.Button(self.frame_diagn_but, height=1, width=1, text = "Traces", fg='black', bg='gray97',
                            command = lambda: self.plot_diagnostic("traces"), font=fontz["normal_small"], relief=tk.GROOVE)
        but_tra.grid(row=6, column=0, sticky=tk.W+tk.E+tk.S, padx=(0,0), pady=(5,0))
        but_his = tk.Button(self.frame_diagn_but, height=1, width=1, text = "Histograms", fg='black', bg='gray97',
                            command = lambda: self.plot_diagnostic("histo"), font=fontz["normal_small"], relief=tk.GROOVE)
        but_his.grid(row=6, column=1, sticky=tk.W+tk.E+tk.S, padx=(5,0), pady=(5,0))
        but_aut = tk.Button(self.frame_diagn_but, height=1, width=1, text = "Autocorrelation", fg='black', bg='gray97',
                            command = lambda: self.plot_diagnostic("autocorr"), font=fontz["normal_small"], relief=tk.GROOVE)
        but_aut.grid(row=6, column=2, sticky=tk.W+tk.E+tk.S, padx=(5,0), pady=(5,0))
        if self.model.get() == "PDecomp":
            but_rtd = tk.Button(self.frame_diagn_but, height=1, text = "Relaxation time distribution", fg='black', bg='gray97',
                                command = self.plot_rtd_now, font=fontz["normal_small"], relief=tk.GROOVE)
            but_rtd.grid(row=5, column=0, columnspan=3, sticky=tk.W+tk.E+tk.S, pady=(5,0))

    def write_output_path(self):
        try:    self.frame_saved_in.destroy()
        except: pass
        self.frame_saved_in = tk.Frame(self.frame_results)
        self.frame_saved_in.grid(row=8, column=0, sticky=tk.E+tk.W, padx=10, pady=0)
        self.frame_saved_in.columnconfigure(0, weight=1)
        label_done = tk.Label(self.frame_saved_in, text="""Results saved in:""", anchor=tk.W)
        label_done.grid(row=0, column=0, columnspan=1, sticky=tk.W+tk.E)
        text_path = tk.Text(self.frame_saved_in, height=3, width=35, font=fontz["normal_small"])
        text_path.grid(row=1, column=0, columnspan=1, sticky=tk.W+tk.E, padx=0)
        if self.run_options["Save CSV results"].get():         
            text_path.insert("1.0", "%s" %(self.working_path))
        else:
            text_path.insert("1.0", "RESULTS NOT SAVED. CSV SAVE OPTION IS UNCHECKED.")
            
    def update_results(self):
        pm = self.all_results[self.var_review.get()]["pm"]
        model = self.model.get()
        try:    self.frame_optimal.destroy()
        except: pass
        self.frame_optimal = tk.Frame(self.frame_results)
        self.frame_optimal.grid(row=2, column=0, sticky=tk.E+tk.W, padx=10, pady=10)
        self.frame_optimal.columnconfigure(0, weight=1)
        keys = sorted([x for x in list(pm.keys()) if "_std" not in x])
        try:
            keys.remove("m_")
        except:
            pass
        if model == "PDecomp":
            adj = -1
        else:
            adj = 0
        values = flatten([pm[k] for k in sorted(keys)])
        errors = flatten([pm[k+"_std"] for k in sorted(keys)])
        for (i, k) in enumerate(keys):
            if len(pm[k].shape) > 0:
                keys[i] = [keys[i]+"%d"%n for n in range(1+adj,pm[k].shape[0]+1+adj)]
        keys = list(flatten(keys))
        self.list_of_parameters = keys
#        for c, k in enumerate(self.list_of_parameters):
#            if "tau" in k:
#                self.list_of_parameters[c] = "log_"+k
        label_res = tk.Label(self.frame_optimal, text="""Optimal parameters:""", anchor=tk.W)
        label_res.grid(row=0, column=0, sticky=tk.W+tk.E+tk.N)
        text_res = tk.Text(self.frame_optimal, height=len(values), width=40, font=("Courier new", fontsize+res_size, "bold"))
        text_res.grid(row=1, column=0, sticky=tk.W+tk.E+tk.N)
        items = ["{:<13}".format(x+":") for x in keys]
        items2 = [" %.3e " %x for x in values]
        items3 = ["+/- %.0e" %x for x in errors]
        items4 = [" (%.2f%%)" %(abs(100*e/v)) for v,e in zip(values,errors)]
        all_items = [a_+b_+c_+d_ for a_,b_,c_,d_ in zip(items,items2,items3,items4)]
        items = '\n'.join(all_items)
        text_res.insert("1.0", items)

#==============================================================================
# MCMC parameters
#==============================================================================
    def mcmc_parameters(self):
        # McMC parameters
        self.adaptive = tk.BooleanVar()
        self.adaptive_text = tk.StringVar()
        def yesno():
            if self.adaptive.get():
                self.adaptive_text.set("AM")
                for i, (k, v) in enumerate(self.mcmc_vars.items()):
                    v[0].set(self.root_ini[k])
                    if k in ["Tuning interval", "Proposal scale"]:
                        tk.Entry(self.frame_mcmc ,textvariable=v[0],fg="grey", width=12).grid(row=i+1, column=1, sticky=tk.E,padx=(0,10))
                        tk.Label(self.frame_mcmc, text=k, justify = tk.LEFT, fg="grey").grid(row=i+1, column=0, sticky=tk.W, padx=(10,0))
                    else:
                        tk.Entry(self.frame_mcmc ,textvariable=v[0], width=12).grid(row=i+1, column=1, sticky=tk.E,padx=(0,10))
                        tk.Label(self.frame_mcmc, text=k, justify = tk.LEFT).grid(row=i+1, column=0, sticky=tk.W, padx=(10,0))
            else:
                self.adaptive_text.set("MH")
                for i, (k, v) in enumerate(self.mcmc_vars.items()):
                    v[0].set(self.root_ini[k])
                    if k in ["Covariance interval", "Covariance delay"]:
                        tk.Entry(self.frame_mcmc ,textvariable=v[0],fg="grey", width=12).grid(row=i+1, column=1, sticky=tk.E,padx=(0,10))
                        tk.Label(self.frame_mcmc, text=k, justify = tk.LEFT, fg="grey").grid(row=i+1, column=0, sticky=tk.W, padx=(10,0))
                    else:
                        tk.Entry(self.frame_mcmc ,textvariable=v[0], width=12).grid(row=i+1, column=1, sticky=tk.E,padx=(0,10))
                        tk.Label(self.frame_mcmc, text=k, justify = tk.LEFT).grid(row=i+1, column=0, sticky=tk.W, padx=(10,0))

        tk.Label(self.frame_mcmc, text="Use Adaptive Metropolis:", justify = tk.LEFT).grid(row=0, column=0, sticky=tk.W, padx=(10,0))
        tk.Checkbutton(self.frame_mcmc , variable=self.adaptive, textvariable=self.adaptive_text, width=12, command=yesno).grid(row=0, column=1, sticky=tk.E,padx=(0,10))
        self.adaptive.set(self.root_ini["Adaptive Metropolis"])
        yesno()

#==============================================================================
# SIP model
#==============================================================================
    def model_choice(self):
        # Available models
        models = [("Pelton \nCole-Cole","ColeCole"),
                  ("Dias \nmodel","Dias"),
                  ("Debye / Warburg \ndecomposition","PDecomp"),]
        self.modes_n, self.poly_n, self.c_exp = tk.IntVar(), tk.IntVar(), tk.DoubleVar()
        self.modes_n.set(self.root_ini["Nb modes"]), self.poly_n.set(self.root_ini["Polyn order"]), self.c_exp.set(self.root_ini["Freq dep"])

        ### Model choice
        self.model = tk.StringVar()
        self.model.set(self.root_ini["Spectral IP model"])  # set last used values
        for i, (txt, val) in enumerate(models):
            tk.Radiobutton(self.frame_model, text=txt, justify=tk.LEFT, variable = self.model, command = self.draw_rtd_check, value=val).grid(row=i, column=0, sticky=tk.W+tk.S, padx=(10,0), pady=pad_radio+2)
        self.draw_rtd_check()

    def draw_rtd_check(self):
        try:
            self.mod_opt_frame.destroy()
        except:
            pass
        self.mod_opt_frame = tk.Frame(self.frame_model)
        self.mod_opt_frame.grid(row=0, column=1, rowspan=4)
        self.mod_opt_frame.grid_rowconfigure(4, weight=1)
        if self.model.get() == "PDecomp":
            poly_lab = tk.Label(self.mod_opt_frame, text="""Polyn order""", justify = tk.LEFT, font=fontz["normal_small"])
            poly_lab.grid(row=0, column=1, rowspan=1, sticky=tk.W+tk.N, pady=(0,0), padx=(0,10))
            poly_scale = tk.Scale(self.mod_opt_frame, variable=self.poly_n, width=10, length=70, from_=3, to=6, font=fontz["normal_small"], orient=tk.HORIZONTAL)
            poly_scale.grid(row=1, column=1, rowspan=1, sticky=tk.E+tk.N, padx=(0,10), pady=(0,0))
            exp_lab = tk.Label(self.mod_opt_frame, text="""c exponent""", justify = tk.LEFT, font=fontz["normal_small"])
            exp_lab.grid(row=2, column=1, rowspan=1, sticky=tk.W+tk.N, pady=(0,0), padx=(0,10))
            exp_scale = tk.Scale(self.mod_opt_frame, variable=self.c_exp, width=10, length=70, from_=0.1, to=1.0, resolution=0.05, font=fontz["normal_small"], orient=tk.HORIZONTAL)
            exp_scale.grid(row=3, column=1, rowspan=1, sticky=tk.E+tk.N, padx=(0,10), pady=(0,0))
        if self.model.get() == "ColeCole":
            modes_lab = tk.Label(self.mod_opt_frame, text="""Nb modes""", justify = tk.LEFT, font=fontz["normal_small"])
            modes_lab.grid(row=0, column=1, rowspan=1, sticky=tk.W+tk.N, pady=(0,0), padx=(0,10))
            modes_scale = tk.Scale(self.mod_opt_frame, variable=self.modes_n, width=10, length=70, from_=1, to=3, font=fontz["normal_small"], orient=tk.HORIZONTAL)
            modes_scale.grid(row=1, column=1, rowspan=1, sticky=tk.E+tk.N, padx=(0,10), pady=(0,0))

#==============================================================================
# Run Exit Options
#==============================================================================
    def run_exit(self):
        for i, (k, v) in enumerate(self.run_options.items()):
            tk.Checkbutton(self.frame_ruex, text=k, variable=v).grid(row=i, column=0, padx=(10,0), sticky=tk.W+tk.N+tk.S)
            self.run_options[k].set(self.root_ini[k]) # set last used values
        tk.Button(self.frame_ruex, width=14, text = "Options", fg='black', bg='gray97', font=fontz["bold"], relief=tk.GROOVE,
                            command = self.make_options).grid(row=i+1, column=0, rowspan=1, sticky=tk.N+tk.E+tk.W+tk.S, padx=(10,5), pady=(0,10))
        tk.Button(self.frame_ruex, width=14, height=1, text = "RUN", fg='blue', bg='gray97', font=fontz["bold"], relief=tk.GROOVE,
                            command = self.run_inversion).grid(row=0, column=1, rowspan=2, sticky=tk.N+tk.E+tk.S, padx=(5,10), pady=(5,0), ipadx=0, ipady=0)
        tk.Button(self.frame_ruex, width=14, height=1, text = "EXIT", fg='red', bg='gray97', font=fontz["bold"], relief=tk.GROOVE,
                            command = self.master.destroy).grid(row=2, column=1, rowspan=2, sticky=tk.S+tk.E+tk.N, padx=(5,10), pady=(0,10))

    def make_options(self, from_root=False):
        try: self.top_checkbox.destroy()
        except: pass

        # Select all or none functions
        def select_all(): # self expl
            for v in list(self.save_options.values()):
                v.set(True)
        def select_none(): # self expl
            for v in list(self.save_options.values()):
                v.set(False)
        if not from_root:
            self.top_checkbox = tk.Toplevel(self.master)
            self.top_checkbox.title("Save options")
            frame_checkbox = tk.LabelFrame(self.top_checkbox, text="Check items to save", width=200, height=4, font=fontz["bold"])
            frame_checkbox.grid(row = 0, column=0,sticky=tk.S+tk.W+tk.E+tk.N, padx=10, pady=10)
            tk.Button(frame_checkbox, height=1, width=15, text="Check all", command=select_all, bg='gray97', font=fontz["normal_small"], relief=tk.GROOVE).grid(row=0, column=0, sticky=tk.W, pady=5, padx=10)
            tk.Button(frame_checkbox, height=1, width=15, text="Check none", command=select_none, bg='gray97', font=fontz["normal_small"], relief=tk.GROOVE).grid(row=0, column=1, sticky=tk.E, pady=5, padx=10)
            button = tk.Button(frame_checkbox, height=2, width=20, text="OK", bg='gray97', font=fontz["bold"], command=self.top_checkbox.destroy, relief=tk.GROOVE)
            button.grid(row=len(self.save_options)+2, column=0, columnspan=2, sticky=tk.W+tk.E, pady=(10,10), padx=(20,20))
        for i, (k, v) in enumerate(sorted(self.save_options.items())):
            if from_root:
                self.save_options[k].set(self.root_ini[k])
            if not from_root:
                tk.Checkbutton(frame_checkbox, text=k, variable=v).grid(row=i+1, column=0, sticky=tk.W+tk.N+tk.S,padx=(10,10),pady=(5,0))

#==============================================================================
# Plotting
#==============================================================================
    def set_plot_par(self): # Setting up plotting parameters
        try:
            print("\nLoading plot parameters")
            rcParams.update(iR.plot_par())
            print("Plot parameters successfully loaded")
        except:
            print("Plot parameters not found, using default values")
        stdout.flush()

    def preview_data(self):
        try:
            for i in self.text_files.curselection():
                sel = str(self.open_files[i])
                fn = sel.split("/")[-1].split(".")[0]
                fig_data = iR.plot_data(sel, self.head.get(), self.units.get())
                self.plot_window(fig_data, "Data preview: "+fn)
        except:
            tkinter.messagebox.showwarning("Preview error",
                                     "Can't draw data\nImport and select at least one data file first", parent=self.master)

    def plot_diagnostic(self, which):
        f_n = self.var_review.get()
        sol = self.all_results[f_n]["sol"]
        try:
            if which == "traces":
                trace_plot = iR.plot_traces(sol, save=False)
                self.plot_window(trace_plot, "Parameter traces: "+f_n)
            if which == "histo":
                histo_plot = iR.plot_histo(sol, save=False)
                self.plot_window(histo_plot, "Parameter histograms: "+f_n)
            if which == "autocorr":
                autocorr_plot = iR.plot_autocorr(sol, save=False)
                self.plot_window(autocorr_plot, "Parameter autocorrelation: "+f_n)
            if which == "geweke":
                geweke_plot = iR.plot_scores(sol, save=False)
                self.plot_window(geweke_plot, "Geweke scores: "+f_n)
            if which == "summary":
                summa_plot = iR.plot_summary(sol, save=False)
                self.plot_window(summa_plot, "Parameter summary: "+f_n)
            if which == "deviance":
                devi_plot = iR.plot_deviance(sol, save=False)
                self.plot_window(devi_plot, "Model deviance: "+f_n)
            if which == "logp":
                logp_plot = iR.plot_logp(sol, save=False)
                self.plot_window(logp_plot, "Log-likelihood: "+f_n)
            if which == "hexbin":
                try: self.top_bivar.destroy()
                except: pass
                hex_plot = iR.plot_hexbin(sol, self.biv1.get(), self.biv2.get(), save=False)
                self.plot_window(hex_plot, "Hexagonal binning: "+f_n)
            if which == "KDE":
                try: self.top_bivar.destroy()
                except: pass
                kde_plot = iR.plot_KDE(sol, self.biv1.get(), self.biv2.get(), save=False)
                self.plot_window(kde_plot, "Bivariate KDE: "+f_n)
            stdout.flush()
        except:
            tkinter.messagebox.showwarning("Error analyzing results", "Error\nProblem with inversion results\nTry adding iterations",
                                     parent=self.master)

    def plot_window(self, fig, name=""):
        top_fig = tk.Toplevel()
        top_fig.lift()
        top_fig.title(name)
        top_fig.rowconfigure(1, weight=1)
        def _quit():
            top_fig.destroy()
        canvas = FigureCanvasTkAgg(fig, master=top_fig)
        canvas.show()
        canvas.get_tk_widget().grid(row=1, column=0, columnspan = 3, pady=(15,15), padx=(25,25), sticky=tk.N+tk.S+tk.E+tk.W)
        button = tk.Button(top_fig, height=1, width=20, text="Dismiss", bg='gray97', command=_quit, relief=tk.GROOVE)
        button.grid(row=2, column=0, columnspan=1, sticky=tk.W, pady=(10,10), padx=(20,20))
        toolbar_frame = tk.Frame(top_fig)
        toolbar_frame.grid(row=0,column=0,columnspan=2, sticky=tk.W+tk.E)
        NavigationToolbar2TkAgg( canvas, toolbar_frame )
        top_fig.resizable(width=tk.FALSE, height=tk.FALSE)

    def plot_fit_now(self):
        f_n = self.var_review.get()
        sol = self.all_results[f_n]["sol"]
        fig_fit = iR.plot_fit(sol, save=False, draw=True)
        self.plot_window(fig_fit, "Inversion results: "+f_n)

    def plot_rtd_now(self):
        f_n = self.var_review.get()
        fig_debye = iR.plot_debye(self.all_results[f_n]["sol"], save=False, draw=True)
        self.plot_window(fig_debye, "Debye RTD: "+f_n)

#==============================================================================
# Importing data
#==============================================================================

    def make_browse_button(self):
        self.but_browse = tk.Button(self.frame_import, height=1, text = "Open file browser", bg='gray97',
                            command = self.get_file_list, relief=tk.GROOVE)
        self.but_browse.grid(row=0, column=0, columnspan=2, pady=(10,10), padx=(10,10), sticky=tk.E+tk.W)

    def headers_input(self):
        self.head = tk.IntVar()
        self.head.set(self.root_ini["Nb header lines"])  # set last used value
        tk.Label(self.frame_import, text="""Nb header lines:""", justify = tk.LEFT).grid(row=1, column=0, columnspan=1,sticky=tk.W, padx=(10,0))
        tk.Entry(self.frame_import, textvariable=self.head, width=12).grid(row=1, column=1, columnspan=1,sticky=tk.E+tk.W,padx=(10,10))

    def phase_units(self):
        # Phase units
        unites = ["mrad", "rad" ,"deg"]
        self.units = tk.StringVar()
        self.units.set(self.root_ini["Phase units"])  # set last used value
        tk.Label(self.frame_import, text="""Phase units:""", justify = tk.LEFT).grid(row=2, column=0, sticky=tk.W, padx=(10,0))
        for i, u in enumerate(unites):
            tk.Radiobutton(self.frame_import, text=u, variable=self.units, command=None, value=u).grid(row=i+2, column=1, sticky=tk.W, padx=(10,10), pady=pad_radio)

    def get_file_list(self):
        files = tkinter.filedialog.askopenfilenames(parent=self.master,title='Choose a file')
        self.open_files = sorted(list(files))
        self.draw_file_list()

    def draw_file_list(self):

        self.frame_path = tk.Frame(self.frame_list)
        self.frame_path.grid(row=0, column=0, padx=20, pady=10, sticky=tk.E+tk.W)
        self.frame_path.columnconfigure(2, weight=1)

        self.frame_select = tk.Frame(self.frame_list)
        self.frame_select.grid(row=1, column=0, padx=(20,0), pady=10, sticky=tk.E+tk.W+tk.S+tk.N)
        self.frame_select.columnconfigure(0, weight=1)
        self.frame_select.rowconfigure(1, weight=1)

        text_loc = tk.Text(self.frame_path, width=40, height=3, font=fontz["normal_small"])
        text_loc.grid(row=0, column=0, rowspan=1, sticky=tk.N+tk.W+tk.E)
        s = tk.Scrollbar(self.frame_select, width=20)
        s.grid(row=1, column=2, sticky=tk.E+tk.N+tk.S, pady=(0,10))
        self.text_files = tk.Listbox(self.frame_select, selectmode='extended', font=fontz["italic_small"])
        self.text_files.grid(row=1, column=0, columnspan=2, sticky=tk.S+tk.W+tk.E+tk.N, pady=(0,0))
        s['command'] = self.text_files.yview
        self.text_files['yscrollcommand'] = s.set

        tk.Button(self.frame_select, height=1, text="Preview selected data files", command=self.preview_data, bg='gray97', font=fontz["normal_small"], relief=tk.GROOVE).grid(row=2, column=0, columnspan=2, sticky=tk.W+tk.E)
        # Select all or none functions
        def select_all(): # self expl
            self.text_files.select_set(0, tk.END)
        def select_none(): # self expl
            self.text_files.select_clear(0, tk.END)
        tk.Button(self.frame_select, height=1, width=15, text="Select all", command=select_all, bg='gray97', font=fontz["normal_small"], relief=tk.GROOVE).grid(row=0, column=0, sticky=tk.W+tk.N, pady=(0,0))
        tk.Button(self.frame_select, height=1, width=15, text="Select none", command=select_none, bg='gray97', font=fontz["normal_small"], relief=tk.GROOVE).grid(row=0, column=1, sticky=tk.E+tk.N, pady=(0,0))

        # Fill list box
        if len(self.open_files) >= 1: # Only draw this listbox if files were selected in browser
            for (i, o) in enumerate(self.open_files):
                fil = o.split("/")[-1] # Split full path to keep only name and extension
                self.text_files.insert(i, fil) # Insert items 1 by 1 in loop
            locs = "/".join(self.open_files[0].split("/")[:-1])+"/" # Path to imported files in string format
            text_loc.insert("1.0", locs) # Insert path to empty text box
            self.text_files.select_set(0) # Select first item in list after populating
        else:
            text_loc.insert("1.0", "None imported") # Empty list box if no files selected

#==============================================================================
# Save and load GUI state
#==============================================================================

    def load(self):
        print("\nLoading root_ini from:\n", self.working_path)
        try:
            with open(self.working_path+'root_ini') as f:
                self.root_ini = jload(f)
            print("root_ini successfully loaded")
        except:
            print("root_ini not found, using default values")
            self.root_ini = self.use_default_root_ini()
        stdout.flush()

    def save(self):
        root_save = {
                        "Spectral IP model"     : self.model.get(),
                        "Adaptive Metropolis"   : self.adaptive.get(),
                        "Nb header lines"       : self.head.get(),
                        "Phase units"           : self.units.get(),
                        "Imported files"        : self.open_files,
                        "Polyn order"           : self.poly_n.get(),
                        "Freq dep"              : self.c_exp.get(),
                        "Nb modes"              : self.modes_n.get(),
                    }

        for k, v in list(self.mcmc_vars.items()):
            root_save[k] = v[0].get()
        for k, v in list(self.save_options.items()):
            root_save[k] = v.get()
        for k, v in list(self.run_options.items()):
            root_save[k] = v.get()

        print("\nSaving root_ini")
        with open(self.working_path+'root_ini', 'w') as f:
            jdump(root_save, f)
        print("root_ini successfully saved in:\n", self.working_path)
        print("\n=========END=========")

#==============================================================================
# Build menubar File, Help
#==============================================================================
    def build_helpmenu(self):
        menubar = tk.Menu(self.master)
        filemenu = tk.Menu(menubar, tearoff=0)
        helpmenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Open", command=self.get_file_list)
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=self.master.destroy)
        menubar.add_cascade(label="File", menu=filemenu)
        helpmenu.add_command(label="Data file template", command=lambda: TextMessage().popup("Data file template", TextMessage.data_template, size=fontsize))
        helpmenu.add_command(label="MCMC parameters", command=lambda: TextMessage().popup("Markov-chain Monte Carlo parameters", TextMessage.mcmc_info, size=fontsize))
        helpmenu.add_command(label="References", command=lambda: TextMessage().popup("References", TextMessage.references, size=fontsize))
        helpmenu.add_separator()
        helpmenu.add_command(label="License", command=lambda: TextMessage().popup("License information", TextMessage.license_info, size=fontsize))
        helpmenu.add_command(label="About", command=lambda : TextMessage().about(fontz))
        menubar.add_cascade(label="Help", menu=helpmenu)
        self.master.config(menu=menubar)



#==============================================================================
# Class for popup messages
#==============================================================================
class TextMessage(object):

    data_template = """DATA FILE TEMPLATE

Save data in .csv, .txt, .dat, ... extension file
Comma separation between columns is mandatory
Column order is very important
Phase units may be milliradians, radians or degrees
Units are specified in main window
Amplitude units may be Ohm-m or Ohm
A number of header lines may be skipped in the main window
In this example Nb header lines = 1
To skip high-frequencies, increase Nb header lines
Scientific or standard notation is OK
Data must be formatted using the following template:

===============================================================================

Freq (Hz), Res (Ohm-m),  Phase (deg), dRes (Ohm-m), dPhase (deg)
6.000e+03, 1.17152e+05, -2.36226e+02, 1.171527e+01, 9.948376e-02
3.000e+03, 1.22177e+05, -1.46221e+02, 1.392825e+01, 1.134464e-01
1.500e+03, 1.25553e+05, -9.51099e+01, 2.762214e+01, 2.199114e-01
........., ..........., ............, ............, ............
........., ..........., ............, ............, ............
........., ..........., ............, ............, ............
4.575e-02, 1.66153e+05, -1.21143e+01, 1.947314e+02, 1.171115e+00
2.288e-02, 1.67988e+05, -9.36718e+00, 3.306003e+02, 1.9669860+00
1.144e-02, 1.70107e+05, -7.25533e+00, 5.630541e+02, 3.310889e+00

===============================================================================
"""

    mcmc_info = """===================================================================================
Meaning of MCMC parameters:

|---------------------------# Iterations------------------------------------| = 100
|........Burn-in period (discarded iterations)........|---------------------| =  79
         Remaining iterations:                        |---------------------| =  21
         Thinning factor (Discard more chains):       |-..-..-..-..-..-..-..| =   3
         Iterations kept for histogram:                             |-------| =   7

         Tuning interval = 1000: the proposal scale is tuned every 1000 iteration
         Proposal scale: the proposal distribution std = scale*parameter_value
         Adding more chains may help sample the posterior distribution
         More iterations will lead to better fit up until a certain point
         Burn-in to discard the random hypotheses before convergence
         Burn-in period is used for tuning
         Thinning factor has no advantage for convergence or quality of fit

         Burn-in for 90% of iterations
         Use thinning factor to keep ~1000-10000 samples in histograms
         (easier on memory)

===================================================================================
Recommended for data files with ~20 frequencies:

Cole-Cole model:    Good convergence for 1 or 2 modes
                    Slow convergence for 3 modes
                    10 000 - 100 000 iterations for 2 modes
                    May take > 500 000 for 3 modes

Debye model:        Good convergence for 3rd order polynomial
                    100 000 iterations usually converges for 3rd order
                    > 100 000 iterations for 4th order polynomial
                    Slow convergence, sometimes no convergence for 5th order

Shin model:         10 000 - 100 000 iterations usually converges

Dias model:         Fast convergence, weak fits
                    10 000 iterations usually converges

===================================================================================
"""

    references = """Chen, Jinsong, Andreas Kemna, and Susan S. Hubbard. 2008. “A Comparison between
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
Patil, A., D. Huard and C.J. Fonnesbeck. 2010. PyMC: Bayesian Stochastic
    Modelling in Python. Journal of Statistical Software, 35(4), pp. 1-81
Pelton, W. H., W. R. Sill, and B. D. Smith. 1983. Interpretation of Complex
    Resistivity and Dielectric Data — Part 1. Vol 29. Geophysical Transactions.
Pelton, W. H., S. H. Ward, P. G. Hallof, W. R. Sill, and P. H. Nelson. 1978.
    “Mineral Discrimination and Removal of Inductive Coupling with
    Multifrequency IP.” Geophysics 43 (3): 588–609. doi:10.1190/1.1440839."""

    license_info = """The MIT License (MIT)

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
"""

    def popup(self, title, message, size=10):
        top = tk.Toplevel()
        top.title(title)
        about_message = (message)
        top.lift()
        msg = tk.Text(top, width=90, font=('courier', size, 'normal'))
        msg.grid(stick=tk.N, padx=(10,10), pady=(10,10))
        msg.insert("1.0", about_message)
        button = tk.Button(top, height=1, width=20, text="Dismiss", command=top.destroy, bg='gray97', relief=tk.GROOVE)
        button.grid(sticky=tk.S, pady=(0,10))
        s = tk.Scrollbar(top, width=20)
        s.grid(row=0, column=0, sticky=tk.E+tk.N+tk.S, padx=(0,10),pady=(10,10))
        s['command'] = msg.yview
        msg['yscrollcommand'] = s.set
        top.resizable(width=tk.FALSE, height=tk.FALSE)

    def about(self, fontz):
        top = tk.Toplevel()
        top.title("About")
        top.lift()
        tk.Label(top,
                  text="""Bayesian inversion of spectral induced polarization data""",
                  justify = tk.CENTER, font=fontz["bold"]).grid(row=0, column=0,columnspan=2,pady=(10,10), padx=(10,10))
        tk.Label(top,
                  text="""2015-2016
    École Polytechnique de Montréal
    Groupe de recherche en géophysique appliquée
    """,
                  justify = tk.CENTER).grid(row=1, column=0,columnspan=2, pady=(10,10), padx=(10,10))
        tk.Label(top,
                  text="""Contact:""",
                  justify = tk.CENTER).grid(row=2, column=0,columnspan=1, pady=(10,10), padx=(10,10))
        about_message = """cberube@ageophysics.com"""
        msg = tk.Text(top, height=1, width=27)
        msg.grid(row=2,column=1,padx=(10,10), pady=(10,10))
        msg.insert("1.0", about_message)
        button = tk.Button(top, height=1, width=20, text="Dismiss", command=top.destroy, bg='gray97', relief=tk.GROOVE)
        button.grid(row=3,column=0,columnspan=2,sticky=tk.S, pady=(0,10))
        top.resizable(width=tk.FALSE, height=tk.FALSE)

#==============================================================================
# Main function
#==============================================================================
def launch():
    root = tk.Tk()
    root.wm_title("Bayesian inversion of SIP data")
    root.option_add("*Font", window_font)
    #==============================================================================
    # For MacOS, bring the window to front
    # Without these lines the application will start in background
    root.lift()
    if "Darwin" in system():
        root.call('wm', 'attributes', '.', '-topmost', True)
        root.after_idle(root.call, 'wm', 'attributes', '.', '-topmost', False)
    app = MainApplication(root, fontz)
    root.mainloop()
    app.save()
if __name__ == '__launch__':
    launch()