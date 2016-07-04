# -*- coding: utf-8 -*-

"""
Created on Tue Apr 21 12:05:22 2015

@author:    clafreniereberube@gmail.com
            École Polytechnique de Montréal

Copyright (c) 2015-2016 Charles L. Bérubé

This python script builds the graphical user interface may be used to call the
Bayesian inversion module for SIP models (BISIP_models.py)
"""

#==============================================================================
# System imports
from sys import argv, stdout
print "\nLoading Python modules"
stdout.flush()
from os.path import dirname as osp_dirname, realpath as osp_realpath
from platform import system
from json import load as jload, dump as jdump
from warnings import filterwarnings
filterwarnings('ignore') # Ignore some tkinter warnings
# GUI imports
import Tkinter as tk
import tkFileDialog, tkMessageBox
import FileDialog, tkFont # These 2 only to avoid pyinstaller error
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.pyplot import rcParams, close as plt_close
# SIP imports
from BISIP_models import mcmcSIPinv
import BISIP_invResults as iR
# Get directory
working_path = str(osp_dirname(osp_realpath(argv[0]))).replace("\\", "/")+"/"
print "All modules successfully loaded"
stdout.flush()
#==============================================================================
# Fonts
if "Darwin" in system():
    size = 12
    pad_radio = 3
    but_size = -2
    res_size = -1
else:
    size = 10
    pad_radio = 0
    but_size = -2
    res_size = -2
window_font = "TkDefaultFont %s"%size
fontz = {"bold": ("TkDefaultFont", size, "bold"),
         "normal_small": ("TkDefaultFont", size+but_size, "normal"),
         "italic_small": ("TkDefaultFont", size+but_size, "italic")}

#==============================================================================
# Import last used window parameters
# All GUI options and choices saved when closing the main window
# Parameters are saved in root_ini file in the executable's directory
# To reset to default parameters, delete root_ini
def load_window_parameters():
    print "\nLoading root_ini from:\n", working_path
    try:
        with open(working_path+'root_ini') as f:
            root_ini = jload(f)
        open_files = root_ini["Imported files"]
        print "root_ini successfully loaded"
    except:
        print "root_ini not found, using default values"
        open_files = []
        itera = 10000
        root_ini = {
            "Spectral IP model" : "ColeCole",
            "Nb of chains:"     : 1,
            "Total iterations:" : itera,
            "Burn-in period:"   : int(itera*0.80),
            "Thinning factor:"  : 1,
            "Tuning interval:"  : 1000,
            "Proposal scale:"   : 1.0,
            "Nb header lines"   : 1,
            "Phase units"       : "mrad",
            "Keep txt traces"   : False,
            "Draw data and fit" : False,
            "Save QC figures "  : False,
            "Save fit figure"   : False,
            "Tuning verbose"    : False,
            "Imported files"    : open_files,
            "Polyn order"       : 4,
            "Nb modes"          : 2,
        }
    stdout.flush()
    return root_ini, open_files

def save_window_parameters():
    root_save = {
        "Spectral IP model" : mod.get(),
        "Nb of chains:"     : mcmc_vars[0][1].get(),
        "Total iterations:" : mcmc_vars[1][1].get(),
        "Burn-in period:"   : mcmc_vars[2][1].get(),
        "Thinning factor:"  : mcmc_vars[3][1].get(),
        "Tuning interval:"  : mcmc_vars[4][1].get(),
        "Proposal scale:"   : mcmc_vars[5][1].get(),
        "Nb header lines"   : head.get(),
        "Phase units"       : uni.get(),
        "Keep txt traces"   : check_vars[3][1].get(),
        "Draw data and fit" : check_vars[0][1].get(),
        "Save QC figures "  : check_vars[2][1].get(),
        "Save fit figure"   : check_vars[1][1].get(),
        "Tuning verbose"    : check_vars[4][1].get(),
        "Imported files"    : open_files,
        "Polyn order"       : poly_n.get(),
        "Nb modes"          : modes_n.get(),
    }
    print "\nSaving root_ini"
    with open(working_path+'root_ini', 'w') as f:
        jdump(root_save, f)
    print "root_ini successfully saved in:\n", working_path
    print "\n=========END========="

# Close the main window
def close_window():
    plt_close("all")
    root.destroy()

# To flatten ND
def flatten(x):
    result = []
    for el in x:
        if hasattr(el, "__iter__") and not isinstance(el, basestring):
            result.extend(flatten(el))
        else:
            result.append(el)
    return result

#==============================================================================
# Block for main inversion function call
def run_inversion():
    global sel_files
    try:    clear(menu=False)
    except: pass
    def all_same(items):
        return all(x == items[0] for x in items)
    sel_files = [str(open_files[i]) for i in text_files.curselection()]
    if len(sel_files) == 0:
        tkMessageBox.showwarning("Inversion error",
                                 "No data selected for inversion \nSelect at least one data file in the left panel", parent=root)
    if len(sel_files) >= 1:
        try:
            Inversion()
            stdout.flush()
        except:
            tkMessageBox.showerror("Inversion error", "Error\nMake sure all fields are OK\nMake sure data file is correctly formatted",
                                         parent=root)
    return

def Inversion():
    global sol, all_results
    model, units, headers = mod.get(), uni.get(), head.get()
    print "\n"
    print "====================="
    print "Starting inversion..."
    print "====================="
    print "Model:", model
    print "Units:", units
    print "Paths:"
    for i in sel_files:
        print i
    print "Skipping", headers, "header lines"
    files = [sel_files[i].split("/")[-1].split(".")[0] for i in range(len((sel_files)))]
    mcmc_params = {"nb_chain"   : mcmc_vars[0][1].get(),
                   "nb_iter"    : mcmc_vars[1][1].get(),
                   "nb_burn"    : mcmc_vars[2][1].get(),
                   "thin"       : mcmc_vars[3][1].get(),
                   "tune_inter" : mcmc_vars[4][1].get(),
                   "prop_scale" : mcmc_vars[5][1].get(),
                   "verbose"    : check_vars[4][1].get(),
                    }
    # Appel de la fonction d'inversion avec les paramètres sélectionnés
    try:    del(all_results)
    except: pass
    all_results = {}
    drop_down(files, 0)
    var_review.set("Working...")
    for (i, f_n) in enumerate(files):
        print "====================="
        activity(f_n,files.index(f_n),len(files),done=False)
        var_review.set(f_n)

        sol = mcmcSIPinv(model, sel_files[i], mcmc= mcmc_params,
                         headers=headers, ph_units=units,
                         cc_modes=modes_n.get(), debye_poly=poly_n.get(),
                         keep_traces=check_vars[3][1].get())

        all_results[f_n] = {"pm":sol["params"],"MDL":sol["pymc_model"],"data":sol["data"],"fit":sol["fit"]}
#         Impression ou non des résultats, graphiques, histogrammes
        update_results(sol["params"], sol["pymc_model"], model)
        iR.print_resul(sol["params"], model, f_n)
        iR.save_resul(sol["pymc_model"], sol["params"], model, f_n)
        fig_fit = iR.plot_fit(sol["data"], sol["fit"], model, f_n, save=check_vars[1][1].get(), draw=check_vars[0][1].get())
        if check_vars[0][1].get():
            plot_window(fig_fit, "Inversion results: "+f_n)
        if model == "Debye":
            iR.plot_debye(sol, f_n, save=check_vars[1][1].get(), draw=False)
        if model == "BDebye":
            iR.plot_debye_histo(sol, f_n, save=check_vars[1][1].get(), draw=False)
        if check_vars[2][1].get():
            iR.plot_histo(all_results[f_n]["MDL"], model, f_n, save=True)
            iR.plot_traces(all_results[f_n]["MDL"], model, f_n, save=True)
            iR.plot_summary(all_results[f_n]["MDL"], model, f_n, mcmc_vars[0][1].get(), save=True)
            iR.plot_autocorr(all_results[f_n]["MDL"], model, f_n, save=True)
        if files.index(f_n)+1 == len(files):
            activity(f_n,files.index(f_n),len(files),done=True)
            diagn_buttons()
            write_output_path()
            print "====================="
            if len(files) > 1:
                iR.merge_results(model,files)
                print "====================="
    return all_results

#==============================================================================
# Block for various plotting functions
def set_plot_par(): # Setting up plotting parameters
    try:
        print "\nLoading plot parameters"
        rcParams.update(iR.plot_par())
        print "Plot parameters successfully loaded"
    except:
        print "Plot parameters not found, using default values"
    stdout.flush()

def preview_data():
    try:
        for i in text_files.curselection():
            sel = str(open_files[i])
            fn = sel.split("/")[-1].split(".")[0]
            fig_data = iR.plot_data(sel, head.get(), uni.get())
            plot_window(fig_data, "Data preview: "+fn)
    except:
        tkMessageBox.showwarning("Preview error",
                                 "Can't draw data\nImport and select at least one data file first", parent=root)

def plot_diagnostic(which):
    filename = var_review.get()
    try:
        if which == "traces":
            trace_plot = iR.plot_traces(all_results[filename]["MDL"], mod.get(), filename, save=False)
            plot_window(trace_plot, "Parameter traces: "+filename)
        if which == "histo":
            histo_plot = iR.plot_histo(all_results[filename]["MDL"], mod.get(), filename, save=False)
            plot_window(histo_plot, "Parameter histograms: "+filename)
        if which == "autocorr":
            autocorr_plot = iR.plot_autocorr(all_results[filename]["MDL"], mod.get(), filename, save=False)
            plot_window(autocorr_plot, "Parameter autocorrelation: "+filename)
        if which == "geweke":
            geweke_plot = iR.plot_scores(all_results[filename]["MDL"], mod.get(), filename, save=False)
            plot_window(geweke_plot, "Geweke scores: "+filename)
        if which == "summary":
            summa_plot = iR.plot_summary(all_results[filename]["MDL"], mod.get(), filename, mcmc_vars[0][1].get(), save=False)
            plot_window(summa_plot, "Parameter summary: "+filename)
        stdout.flush()
    except:
        tkMessageBox.showwarning("Error analyzing results", "Error\nProblem with inversion results\nTry adding iterations",
                                 parent=root)

def diagnostic():
    MDL = all_results[var_review.get()]["MDL"]
    if mod.get() == "Debye": adj = -1
    else: adj = 0
    try:
        keys = [x.__name__ for x in MDL.stochastics]
        for (i, k) in enumerate(keys):
            vect = (MDL.trace(k)[:].size)/(len(MDL.trace(k)[:]))
            if vect > 1:
             keys[i] = [k+"%d"%n for n in range(1+adj,vect+1+adj)]
        keys = list(reversed(sorted(flatten(keys))))
        top_RLD = tk.Toplevel()
        top_RLD.title("Raftery-Lewis diagnostic")
        text_RLD = tk.Text(top_RLD, width=110, font=('courier', size, 'normal'))
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
        text_RLD.insert("1.0", var_review.get()+"\n\n")
        button = tk.Button(top_RLD, height=1, width=20, text="Dismiss", command=top_RLD.destroy, bg='gray97', relief=tk.GROOVE)
        button.grid(row=1, column=0, sticky=tk.S, pady=(0,10))

        s = tk.Scrollbar(top_RLD, width=20)
        s.grid(row=0, column=1, sticky=tk.E+tk.N+tk.S, padx=(0,0),pady=(10,10))
        s['command'] = text_RLD.yview
        text_RLD['yscrollcommand'] = s.set
        top_RLD.resizable(width=tk.FALSE, height=tk.FALSE)
    except:
        tkMessageBox.showwarning("Diagnostic error",
                                 "Error\nRun inversion first", parent=root)

def plot_fit_now():
    f_n = var_review.get()
    data = all_results[f_n]["data"]
    fit = all_results[f_n]["fit"]
    fig_fit = iR.plot_fit(data, fit, mod.get(), f_n, save=False, draw=True)
    print f_n
    plot_window(fig_fit, "Inversion results: "+f_n)

def plot_rtd_now():
    f_n = var_review.get()
    pm = all_results[f_n]["pm"]
    fig_debye = iR.plot_debye(sol, f_n, save=False, draw=True)
    plot_window(fig_debye, "Debye RTD: "+f_n)

def plot_rtdhisto_now():
    f_n = var_review.get()
    fig_debye = iR.plot_debye_histo(sol, f_n, save=False, draw=True)
    plot_window(fig_debye, "Debye RTD: "+f_n)

def plot_window(fig, name):
    top_fig = tk.Toplevel()
    top_fig.lift()
    top_fig.title(name)
    top_fig.rowconfigure(1, weight=1)
    def _quit():
        top_fig.destroy()
    canvas = FigureCanvasTkAgg(fig, master=top_fig)
    canvas.show()
    canvas.get_tk_widget().grid(row=1, column=0, columnspan = 3, sticky=tk.N+tk.S+tk.E+tk.W)
    button = tk.Button(top_fig, height=1, width=20, text="Dismiss", bg='gray97', command=_quit, relief=tk.GROOVE)
    button.grid(row=2, column=0, columnspan=1, sticky=tk.W, pady=(10,10), padx=(20,20))
    toolbar_frame = tk.Frame(top_fig)
    toolbar_frame.grid(row=0,column=0,columnspan=2, sticky=tk.W)
    NavigationToolbar2TkAgg( canvas, toolbar_frame )
    top_fig.resizable(width=tk.FALSE, height=tk.FALSE)

#==============================================================================
# Block for browsing, importing, selecting data files
def browse():
    but_browse = tk.Button(frame_import, height=1, text = "Open file browser", bg='gray97',
                        command = get_file_list, relief=tk.GROOVE)
    but_browse.grid(row=0, column=0, columnspan=2, pady=(10,10), padx=(10,10), sticky=tk.E+tk.W)
    return but_browse

def get_file_list():
    global open_files
    files = tkFileDialog.askopenfilenames(parent=root,title='Choose a file')
    open_files = sorted(list(files))
    draw_file_list(open_files)

def draw_file_list(open_files):
    global text_files, text_loc
    try: text_files.destroy()
    except: pass
    frame_path = tk.Frame(frame_list)
    frame_path.grid(row=0, column=0, padx=20, pady=10, sticky=tk.E+tk.W)
    frame_path.columnconfigure(2, weight=1)

    frame_select = tk.Frame(frame_list)
    frame_select.grid(row=1, column=0, padx=(20,0), pady=10, sticky=tk.E+tk.W+tk.S+tk.N)
    frame_select.columnconfigure(0, weight=1)
    frame_select.rowconfigure(1, weight=1)

    text_loc = tk.Text(frame_path, width=40, height=3, font=fontz["normal_small"])
    text_loc.grid(row=0, column=0, rowspan=1, sticky=tk.N+tk.W+tk.E)
    s = tk.Scrollbar(frame_select, width=20)
    s.grid(row=1, column=2, sticky=tk.E+tk.N+tk.S, pady=(0,10))
    text_files = tk.Listbox(frame_select, selectmode='extended', font=fontz["italic_small"])
    text_files.grid(row=1, column=0, columnspan=2, sticky=tk.S+tk.W+tk.E+tk.N, pady=(0,0))
    s['command'] = text_files.yview
    text_files['yscrollcommand'] = s.set
    tk.Button(frame_select, height=1, text="Preview selected data files", command=preview_data, bg='gray97', font=fontz["normal_small"], relief=tk.GROOVE).grid(row=2, column=0, columnspan=2, sticky=tk.W+tk.E)
    # Select all or none functions
    def select_all(): # self expl
        text_files.select_set(0, tk.END)
    def select_none(): # self expl
        text_files.select_clear(0, tk.END)
    tk.Button(frame_select, height=1, width=15, text="Select all", command=select_all, bg='gray97', font=fontz["normal_small"], relief=tk.GROOVE).grid(row=0, column=0, sticky=tk.W+tk.N, pady=(0,0))
    tk.Button(frame_select, height=1, width=15, text="Select none", command=select_none, bg='gray97', font=fontz["normal_small"], relief=tk.GROOVE).grid(row=0, column=1, sticky=tk.E+tk.N, pady=(0,0))
    # Fill list box
    if len(open_files) >= 1: # Only draw this listbox if files were selected in browser
        for (i, o) in enumerate(open_files):
            fil = o.split("/")[-1] # Split full path to keep only name and extension
            text_files.insert(i, fil) # Insert items 1 by 1 in loop
        locs = "/".join(open_files[0].split("/")[:-1])+"/" # Path to imported files in string format
        text_loc.insert("1.0", locs) # Insert path to empty text box
        text_files.select_set(0) # Select first item in list after populating
    else:
        text_loc.insert("1.0", "None imported") # Empty list box if no files selected

#==============================================================================
# Block for results frames
def write_output_path():
    global frame_saved_in
    frame_saved_in = tk.Frame(frame_results)
    frame_saved_in.grid(row=8, column=0, sticky=tk.E+tk.W, padx=10, pady=0)
    frame_saved_in.columnconfigure(0, weight=1)
    label_done = tk.Label(frame_saved_in, text="""Results saved in:""", anchor=tk.W)
    label_done.grid(row=0, column=0, columnspan=1, sticky=tk.W+tk.E)
    text_path = tk.Text(frame_saved_in, height=3, width=35, font=fontz["normal_small"])
    text_path.grid(row=1, column=0, columnspan=1, sticky=tk.W+tk.E, padx=0)
    text_path.insert("1.0", "%s" %(working_path))

def update_results(pm, MDL, model):
    global all_items, text_res, label_res
    global frame_optimal
    try:    frame_optimal.destroy()
    except: pass
    frame_optimal = tk.Frame(frame_results)
    frame_optimal.grid(row=2, column=0, sticky=tk.E+tk.W, padx=10, pady=10)
    frame_optimal.columnconfigure(0, weight=1)
    keys = sorted([x for x in pm.keys() if "_std" not in x])
    if model == "Debye":
        adj = -1
        if "m" in keys: keys.remove("m")
    else:   adj = 0
    values = flatten([pm[k] for k in sorted(keys)])
    errors = flatten([pm[k+"_std"] for k in sorted(keys)])
    for (i, k) in enumerate(keys):
        if len(pm[k].shape) > 0:
            keys[i] = [keys[i]+"%d"%n for n in range(1+adj,pm[k].shape[0]+1+adj)]
    keys = list(flatten(keys))
    label_res = tk.Label(frame_optimal, text="""Optimal parameters:""", anchor=tk.W)
    label_res.grid(row=0, column=0, sticky=tk.W+tk.E+tk.N)
    text_res = tk.Text(frame_optimal, height=len(values), width=40, font=("Courier new", size+res_size, "bold"))
    text_res.grid(row=1, column=0, sticky=tk.W+tk.E+tk.N)
    items = ["{:<13}".format(x+":") for x in keys]
    items2 = [" %.3e " %x for x in values]
    items3 = ["+/- %.0e" %x for x in errors]
    items4 = [" (%.2f%%)" %(abs(100*e/v)) for v,e in zip(values,errors)]
    all_items = map(str.__add__,map(str.__add__,map(str.__add__,items,items2),items3),items4)
    items = '\n'.join(all_items)
    text_res.insert("1.0", items)

# Batch progress
def activity(fn,f,nf,done):
    global frame_activ
    try:    frame_activ.destroy()
    except: pass
    frame_activ = tk.Frame(frame_results)
    frame_activ.grid(row=0, column=0, sticky=tk.E+tk.W, padx=10, pady=10)
    if (fn, f, nf) == (None,None,None):
        display = """ \n """
        text_act = tk.Label(frame_activ, text=display, anchor=tk.W, justify=tk.LEFT, width=40)
    else:
        display = """Working on:\n%s (#%d/%d)..."""%(fn,f+1,nf)
        text_act = tk.Label(frame_activ, text=display, anchor=tk.W, justify=tk.LEFT, width=40)
    text_act.grid(row=0, column=0, sticky=tk.W+tk.E+tk.N)
    text_act.update()

    if done:
        text_act.destroy()
        text_done = tk.Label(frame_activ, text="""Done\n""", width=40, anchor=tk.W)
        text_done.grid(row=0, column=0, sticky=tk.W+tk.E+tk.N)

# Drop down menu
def drop_down(files,val):
    global var_review, frame_drop
    try:    frame_drop.destroy()
    except: pass
    frame_drop = tk.Frame(frame_results)
    frame_drop.grid(row=1, column=0, sticky=tk.E+tk.W, padx=10, pady=3)
    frame_drop.columnconfigure(0,weight=1)
    var_review = tk.StringVar()
    var_review.set(files[val])
    def change_file(v,mdl,m):
        clear(menu=False)
        update_results(v,mdl,m)
        diagn_buttons()
        write_output_path()
    optionmenu = tk.OptionMenu(frame_drop, var_review, *files, command=lambda x: change_file(all_results[var_review.get()]["pm"],all_results[var_review.get()]["MDL"],mod.get()))
    optionmenu.grid(row=0, column=0, sticky=tk.W+tk.E+tk.S)
    optionmenu.config(bg = "gray97", relief=tk.GROOVE)

# Clear button
def clear(menu): # self expl
    frame_optimal.destroy()
    frame_saved_in.destroy()
    frame_diagn_but.destroy()
    if menu:
        frame_drop.destroy(), frame_activ.destroy()

# Diagnostics buttons
def diagn_buttons():
    global frame_diagn_but
    try:    frame_diagn_but.destroy()
    except: pass
    frame_diagn_but = tk.Frame(frame_results)
    frame_diagn_but.grid(row=14, column=0, sticky=tk.E+tk.W+tk.S, padx=10, pady=10)
    frame_diagn_but.columnconfigure((0,1,2), weight=1)
    but_cle = tk.Button(frame_diagn_but, height=1, width=10, text = "Clear", fg='black', bg='gray97', font=fontz["normal_small"],
                        command = lambda: clear(menu=True), relief=tk.GROOVE)
    but_cle.grid(row=6, column=0, columnspan=3, sticky=tk.E+tk.S, pady=(5,0))
    but_fit = tk.Button(frame_diagn_but, height=1, text = "Draw data and fit", fg='black', bg='gray97',
                        command = plot_fit_now, font=fontz["normal_small"], relief=tk.GROOVE)
    but_fit.grid(row=1, column=0, columnspan=3, sticky=tk.W+tk.E+tk.S, pady=(5,0))
    but_sum = tk.Button(frame_diagn_but, height=1, text = "Summary and Gelman-Rubin convergence", fg='black', bg='gray97',
                        command = lambda: plot_diagnostic("summary"), font=fontz["normal_small"], relief=tk.GROOVE)
    but_sum.grid(row=4, column=0, columnspan=3, sticky=tk.W+tk.E+tk.S, pady=(5,0))
    but_rld = tk.Button(frame_diagn_but, height=1, text = "Raftery-Lewis diagnostic", fg='black', bg='gray97',
                        command = diagnostic, font=fontz["normal_small"], relief=tk.GROOVE)
    but_rld.grid(row=5, column=0, columnspan=3, sticky=tk.W+tk.E+tk.S, pady=(5,5))
    but_tra = tk.Button(frame_diagn_but, height=1, width=1, text = "Traces", fg='black', bg='gray97',
                        command = lambda: plot_diagnostic("traces"), font=fontz["normal_small"], relief=tk.GROOVE)
    but_tra.grid(row=3, column=0, sticky=tk.W+tk.E+tk.S, padx=(0,0), pady=(5,0))
    but_his = tk.Button(frame_diagn_but, height=1, width=1, text = "Histograms", fg='black', bg='gray97',
                        command = lambda: plot_diagnostic("histo"), font=fontz["normal_small"], relief=tk.GROOVE)
    but_his.grid(row=3, column=1, sticky=tk.W+tk.E+tk.S, padx=(5,0), pady=(5,0))
    but_aut = tk.Button(frame_diagn_but, height=1, width=1, text = "Autocorrelation", fg='black', bg='gray97',
                        command = lambda: plot_diagnostic("autocorr"), font=fontz["normal_small"], relief=tk.GROOVE)
    but_aut.grid(row=3, column=2, sticky=tk.W+tk.E+tk.S, padx=(5,0), pady=(5,0))
#    but_gew = tk.Button(frame_results, height=1, text = "Geweke scores", fg='black',
#                        command = lambda: plot_diagnostic("geweke"))
#    but_gew.grid(row=4, column=0, sticky=tk.W+tk.E+tk.S, padx=10, pady=0)
    if mod.get() == "Debye":
        but_rtd = tk.Button(frame_diagn_but, height=1, text = "Relaxation time distribution", fg='black', bg='gray97',
                            command = plot_rtd_now, font=fontz["normal_small"], relief=tk.GROOVE)
        but_rtd.grid(row=2, column=0, columnspan=3, sticky=tk.W+tk.E+tk.S, pady=(5,0))

    if mod.get() == "BDebye":
        but_rtd = tk.Button(frame_diagn_but, height=1, text = "Relaxation time distribution", fg='black', bg='gray97',
                            command = plot_rtdhisto_now, font=fontz["normal_small"], relief=tk.GROOVE)
        but_rtd.grid(row=2, column=0, columnspan=3, sticky=tk.W+tk.E+tk.S, pady=(5,0))


#==============================================================================
# Block for model frame
def model_choice():
    # Available models
    models = [("Pelton Cole-Cole","ColeCole"),
              ("Dias model","Dias"),
              ("Polynomial Debye","Debye"),
              ("Discrete Debye","BDebye"),]
    modes_n, poly_n = tk.IntVar(), tk.IntVar()
    modes_n.set(root_ini["Nb modes"]), poly_n.set(root_ini["Polyn order"])    # Initial values for sliders in case None given
    def draw_rtd_check():
        global mod_opt_frame
        try:    mod_opt_frame.destroy()
        except: pass
        mod_opt_frame = tk.Frame(frame_model)
        mod_opt_frame.grid(row=0, column=1, rowspan=4)
        mod_opt_frame.grid_rowconfigure(4, weight=1)
        if mod.get() == "Debye":
            poly_lab = tk.Label(mod_opt_frame, text="""Polyn order""", justify = tk.LEFT, font=fontz["normal_small"])
            poly_lab.grid(row=0, column=1, rowspan=1, sticky=tk.W+tk.N, pady=(0,0), padx=(0,10))
            poly_scale = tk.Scale(mod_opt_frame, variable=poly_n, width=10, length=70, from_=3, to=5, font=fontz["normal_small"], orient=tk.HORIZONTAL)
            poly_scale.grid(row=1, column=1, rowspan=1, sticky=tk.E+tk.N, padx=(0,10), pady=(0,0))
        if mod.get() == "ColeCole":
            modes_lab = tk.Label(mod_opt_frame, text="""Nb modes""", justify = tk.LEFT, font=fontz["normal_small"])
            modes_lab.grid(row=0, column=1, rowspan=1, sticky=tk.W+tk.N, pady=(0,0), padx=(0,10))
            modes_scale = tk.Scale(mod_opt_frame, variable=modes_n, width=10, length=70, from_=1, to=3, font=fontz["normal_small"], orient=tk.HORIZONTAL)
            modes_scale.grid(row=1, column=1, rowspan=1, sticky=tk.E+tk.N, padx=(0,10), pady=(0,0))
    ### Model choice
    mod = tk.StringVar()
    mod.set(root_ini["Spectral IP model"])  # set last used values
    for i, (txt, val) in enumerate(models):
        tk.Radiobutton(frame_model, text=txt, variable = mod, command = draw_rtd_check, value=val).grid(row=i, column=0, sticky=tk.W, padx=(10,0), pady=pad_radio+2)
    draw_rtd_check()
    return mod, modes_n, poly_n

#==============================================================================
# Block for entry boxes with MCMC parameters
def mcmc_parameters():
    # McMC parameters
    mcmc_vars = [("Nb of chains:"    , tk.IntVar()),
                 ("Total iterations:", tk.IntVar()),
                 ("Burn-in period:"  , tk.IntVar()),
                 ("Thinning factor:" , tk.IntVar()),
                 ("Tuning interval:" , tk.IntVar()),
                 ("Proposal scale:"  , tk.DoubleVar()),]
    for i, (k, v) in enumerate(mcmc_vars):
        v.set(root_ini[k])
        tk.Label(frame_mcmc, text=k, justify = tk.LEFT).grid(row=i, column=0, sticky=tk.W, padx=(10,0))
        tk.Entry(frame_mcmc ,textvariable=v, width=12).grid(row=i, column=1, sticky=tk.E,padx=(0,10))
    return mcmc_vars

#==============================================================================
# Block for import data frame
def skip_headers():
    # Headers to skip
    head = tk.IntVar()
    head.set(root_ini["Nb header lines"])  # set last used value
    tk.Label(frame_import, text="""Nb header lines:""", justify = tk.LEFT).grid(row=1, column=0, columnspan=1,sticky=tk.W, padx=(10,0))
    tk.Entry(frame_import,textvariable=head, width=12).grid(row=1, column=1, columnspan=1,sticky=tk.E+tk.W,padx=(10,10))
    return head

def phase_units():
    # Phase units
    unites = [("mrad","mrad"),
              ("rad" ,"rad"),
              ("deg" ,"deg"),]
    uni = tk.StringVar()
    uni.set(root_ini["Phase units"])  # set last used value
    tk.Label(frame_import, text="""Phase units:""", justify = tk.LEFT).grid(row=2, column=0, sticky=tk.W, padx=(10,0))
    for i, (txt, val) in enumerate(unites):
        tk.Radiobutton(frame_import, text=txt, variable=uni, command=None, value=val).grid(row=i+2, column=1, sticky=tk.W, padx=(10,10), pady=pad_radio)
    return uni

#==============================================================================
# Block for batch options and run/exit buttons
def checkboxes(): # self expl
    check_vars = [("Draw data and fit", tk.BooleanVar()),
                  ("Save fit figure"  , tk.BooleanVar()),
                  ("Save QC figures " , tk.BooleanVar()),
                  ("Keep txt traces"  , tk.BooleanVar()),
                  ("Tuning verbose"   , tk.BooleanVar()),]
    frame_checkbox = tk.Frame(frame_ruex)
    frame_checkbox.grid(row=0, column=0, rowspan=2, pady=(5,10), padx=10)
    for i, (k, v) in enumerate(check_vars):
        tk.Checkbutton(frame_checkbox, text=k, variable=v).grid(row=i, column=0, sticky=tk.W+tk.N+tk.S)
        check_vars[i][1].set(root_ini[k]) # set last used values
    return check_vars

def run_exit(): # self expl
    tk.Button(frame_ruex, width=14, text = "RUN", fg='blue', bg='gray97', font=fontz["bold"], relief=tk.GROOVE,
                        command = run_inversion).grid(row=0, column=1, sticky=tk.N+tk.E+tk.S, padx=(0,10), pady=(5,0))

    tk.Button(frame_ruex, width=14, text = "EXIT", fg='red', bg='gray97', font=fontz["bold"], relief=tk.GROOVE,
                        command = close_window).grid(row=1, column=1, sticky=tk.S+tk.E+tk.N, padx=(0,10), pady=(0,10))

#==============================================================================
# Main window frames - geometry - font - labels
def main_frames():
    global frame_import, frame_fileopt, frame_model, frame_mcmc, frame_ruex, frame_list, frame_results
    frame_import = tk.LabelFrame(root, text="1. Import data", width=200, height=5, font=fontz["bold"])
    frame_import.grid(row = 0, column=1, columnspan=1, rowspan=5, sticky=tk.W+tk.E+tk.N, padx=10, pady=(5,15))
    frame_import.grid_columnconfigure(0, weight=1), frame_import.grid_rowconfigure(0, weight=1)
    frame_model = tk.LabelFrame(root, text="2. SIP model", width=200, height=4, font=fontz["bold"])
    frame_model.grid(row = 6, column=1, columnspan=1, sticky=tk.W+tk.E+tk.N, padx=10, pady=(5,15))
    frame_model.grid_columnconfigure(0, weight=1), frame_model.grid_rowconfigure(0, weight=1)
    frame_mcmc = tk.LabelFrame(root, text="3. MCMC settings", width=200, height=4, font=fontz["bold"])
    frame_mcmc.grid(row = 7, column=1, columnspan=1, sticky=tk.W+tk.E+tk.N, padx=10, pady=(5,15), ipady=3)
    frame_mcmc.grid_columnconfigure(0, weight=1)
    frame_ruex = tk.LabelFrame(root, text="4. Options", width=200, height=4, font=fontz["bold"])
    frame_ruex.grid(row = 8, column=1, columnspan=1, sticky=tk.S+tk.W+tk.E+tk.N, padx=10, pady=(5,10))
    frame_ruex.columnconfigure(0, weight=1)
    frame_list = tk.LabelFrame(root, text="List of imported files", font=fontz["bold"])
    frame_list.grid(row = 0, column=0, columnspan=1, rowspan=15, sticky=tk.W+tk.E+tk.N+tk.S, padx=10, pady=(5,10))
    frame_list.grid_rowconfigure(1, weight=1), frame_list.columnconfigure(0, weight=1)
    frame_results = tk.LabelFrame(root, text="Results", font=fontz["bold"])
    frame_results.grid(row = 0, column=2, columnspan=1, rowspan=15, sticky=tk.W+tk.E+tk.N+tk.S, padx=10, pady=(5,10))
    frame_results.grid_rowconfigure(14, weight=1)

#==============================================================================
# Build menubar File, Help
def build_helpmenu():
    menubar = tk.Menu(root)
    filemenu = tk.Menu(menubar, tearoff=0)
    helpmenu = tk.Menu(menubar, tearoff=0)
    filemenu.add_command(label="Open", command=get_file_list)
    filemenu.add_separator()
    filemenu.add_command(label="Exit", command=close_window)
    menubar.add_cascade(label="File", menu=filemenu)
    helpmenu.add_command(label="Data file template", command=lambda: example(size))
    helpmenu.add_command(label="MCMC parameters", command=lambda:help_mcmc(size))
    helpmenu.add_command(label="References", command=lambda: references(size))
    helpmenu.add_separator()
    helpmenu.add_command(label="About", command=lambda:about(fontz))
    menubar.add_cascade(label="Help", menu=helpmenu)
    return menubar

def references(size):
    top = tk.Toplevel()
    top.title("References")
    top.lift()
    about_message = """Chen, Jinsong, Andreas Kemna, and Susan S. Hubbard. 2008. “A Comparison between
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

    msg = tk.Text(top, height=30, font=('courier', size, 'normal'))
    msg.grid(stick=tk.N, padx=(10,10), pady=(10,10))
    msg.insert("1.0", about_message)

    button = tk.Button(top, height=1, width=20, text="Dismiss", command=top.destroy, bg='gray97', relief=tk.GROOVE)
    button.grid(sticky=tk.S, pady=(0,10))
    top.resizable(width=tk.FALSE, height=tk.FALSE)

def example(size):
    top = tk.Toplevel()
    top.title("Data file template")
    top.lift()
    about_message = """DATA FILE TEMPLATE

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

=============================================================================

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

=============================================================================
"""
    msg = tk.Text(top, height=31, width=77, font=('courier', size, 'normal'))
    msg.grid(stick=tk.N, padx=(10,10), pady=(10,10))
    msg.insert("1.0", about_message)
    button = tk.Button(top, height=1, width=20, text="Dismiss", command=top.destroy, bg='gray97', relief=tk.GROOVE)
    button.grid(sticky=tk.S, pady=(0,10))
    top.resizable(width=tk.FALSE, height=tk.FALSE)

def help_mcmc(size):
    top = tk.Toplevel()
    top.title("Markov-chain Monte Carlo parameters")
    top.lift()
    about_message = """===================================================================================
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
    msg = tk.Text(top, height=41, width=83, font=('courier', size, 'normal'))
    msg.grid(stick=tk.N, padx=(10,10), pady=(10,10))
    msg.insert("1.0", about_message)

    button = tk.Button(top, height=1, width=20, text="Dismiss", command=top.destroy, bg='gray97', relief=tk.GROOVE)
    button.grid(sticky=tk.S, pady=(0,10))
    top.resizable(width=tk.FALSE, height=tk.FALSE)

def about(fontz):
    top = tk.Toplevel()
    top.title("About")
    top.lift()
    tk.Label(top,
              text="""Bayesian inversion of spectral induced polarization data""",
              justify = tk.CENTER, font=fontz["bold"]).grid(row=0, column=0,columnspan=2,pady=(10,10), padx=(10,10))
    tk.Label(top,
              text="""Copyright (c) 2015-2016 Charles L. Bérubé
École Polytechnique de Montréal
Groupe de recherche en géophysique appliquée
""",
              justify = tk.CENTER).grid(row=1, column=0,columnspan=2, pady=(10,10), padx=(10,10))
    tk.Label(top,
              text="""Contact:""",
              justify = tk.CENTER).grid(row=2, column=0,columnspan=1, pady=(10,10), padx=(10,10))
    about_message = """charleslberube@gmail.com"""
    msg = tk.Text(top, height=1, width=27)
    msg.grid(row=2,column=1,padx=(10,10), pady=(10,10))
    msg.insert("1.0", about_message)
    button = tk.Button(top, height=1, width=20, text="Dismiss", command=top.destroy, bg='gray97', relief=tk.GROOVE)
    button.grid(row=3,column=0,columnspan=2,sticky=tk.S, pady=(0,10))
    top.resizable(width=tk.FALSE, height=tk.FALSE)




#==============================================================================
# Window start
root = tk.Tk()
root.wm_title("Bayesian inversion of SIP data using MCMC")
root.option_add("*Font", window_font)
#==============================================================================
# Build and display menu
root.config(menu=build_helpmenu())
#==============================================================================
# Build GUI by calling the main GUI functions
root_ini, open_files = load_window_parameters()
set_plot_par()
main_frames()
draw_file_list(root_ini["Imported files"])
but_browse = browse()
mod, modes_n, poly_n = model_choice()
mcmc_vars = mcmc_parameters()
head = skip_headers()
uni = phase_units()
check_vars = checkboxes()
run_exit()
activity(None,None,None,None)
#==============================================================================
# Window size options
root.maxsize(width=root.winfo_width(), height=root.winfo_height())
root.minsize(width=root.winfo_width(), height=root.winfo_height())
root.resizable(width=tk.FALSE, height=tk.FALSE)
#==============================================================================
# For MacOS, bring the window to front
# Without these lines the application will start in background
if "Darwin" in system():
    root.lift()
    root.call('wm', 'attributes', '.', '-topmost', True)
    root.after_idle(root.call, 'wm', 'attributes', '.', '-topmost', False)
#==============================================================================
# Window loop
tk.mainloop()
save_window_parameters()
#==============================================================================