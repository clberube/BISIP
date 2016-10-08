# -*- coding: utf-8 -*-

"""
Created on Tue Apr 21 12:05:22 2015

@author:    charleslberube@gmail.com
            École Polytechnique de Montréal

Copyright (c) 2015-2016 Charles L. Bérubé

This python script builds the graphical user interface may be used to call the
Bayesian inversion module for SIP models (BISIP_models.py)
"""

#==============================================================================
# System imports
print "\nLoading Python modules"
from sys import argv, stdout
stdout.flush()
from platform import system
from os.path import dirname as osp_dirname, realpath as osp_realpath
from json import load as jload, dump as jdump
from warnings import filterwarnings
filterwarnings('ignore') # Ignore some tkinter warnings
# GUI imports
import Tkinter as tk
import FixTk
import tkFileDialog, tkMessageBox
import FileDialog, tkFont # These 2 only to avoid pyinstaller error
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.pyplot import rcParams, close as plt_close
# SIP imports
from BISIP_models import mcmcSIPinv
import BISIP_invResults as iR
# Get directory
print "All modules successfully loaded"
stdout.flush()
#==============================================================================
# Fonts
if "Darwin" in system():
    print "OSX detected"
    fontsize = 12
    pad_radio = 3
    but_size = -2
    res_size = -1
else:
    fontsize = 10
    pad_radio = 0
    but_size = -2
    res_size = -2

window_font = "TkDefaultFont %s"%fontsize
fontz = {"bold": ("TkDefaultFont", fontsize, "bold"),
         "normal_small": ("TkDefaultFont", fontsize+but_size, "normal"),
         "italic_small": ("TkDefaultFont", fontsize+but_size, "italic")}

#==============================================================================
# Import last used window parameters
# All GUI options and choices saved when closing the main window
# Parameters are saved in root_ini file in the executable's directory
# To reset and use default parameters, delete root_ini in local directory
#class root_state:
    
class MainApplication:
    def __init__(self, master, fontz):
        self.master = master
        self.master.resizable(width=tk.FALSE, height=tk.FALSE)
        self.load()     
        self.set_plot_par()
        self.make_main_frames()
        self.make_browse_button()
        self.headers_input()
        self.open_files = self.root_ini["Imported files"]
        self.draw_file_list()
        self.phase_units()
        self.mcmc_parameters()
        self.run_exit()
        self.make_options()
        self.model_choice()
        self.activity(idle=True)
        
    def run_inversion(self):
#        try:    clear(menu=False)
#        except: pass
        self.sel_files = [str(self.open_files[i]) for i in self.text_files.curselection()]
        if len(self.sel_files) == 0:
            tkMessageBox.showwarning("Inversion error",
                                     "No data selected for inversion \nSelect at least one data file in the left panel", parent=self.master)
        if len(self.sel_files) >= 1:
            try:
                self.Inversion()
                stdout.flush()
            except:
                tkMessageBox.showerror("Inversion error", "Error\nMake sure all fields are OK\nMake sure data file is correctly formatted",
                                             parent=self.master)
        return

    # Drop down menu
    def draw_drop_down(self):
#        try:    frame_drop.destroy()
#        except: pass
        self.frame_drop = tk.Frame(self.frame_results)
        self.frame_drop.grid(row=1, column=0, sticky=tk.E+tk.W, padx=10, pady=3)
        self.frame_drop.columnconfigure(0,weight=1)
        self.var_review = tk.StringVar()
        self.var_review.set(self.files[0])

        optionmenu = tk.OptionMenu(self.frame_drop, self.var_review, *self.files, command=None)
        optionmenu.grid(row=0, column=0, sticky=tk.W+tk.E+tk.S)
        optionmenu.config(bg = "gray97", relief=tk.GROOVE)


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

    def Inversion(self):
        print "\n"
        print "====================="
        print "Starting inversion..."
        print "====================="
        print "Model:", self.mod.get()
        if self.mod.get() == "ColeCole":
            print "Cole-Cole modes:", self.modes_n.get()
        if self.mod.get() == "PDecomp":
            print "Polynomial order:", self.poly_n.get()
            if self.c_exp.get() == 1.0:
                decomp_type = "(Debye)"
            elif self.c_exp.get() == 0.5:
                decomp_type = "(Warburg)"
            else:
                decomp_type = "(Cole-Cole)"
            print "Frequency dependence:", self.c_exp.get(), decomp_type
        print "Units:", self.units.get()
        print "Paths:"
        for i in self.sel_files: 
            print i
        print "Skipping", self.head.get(), "header lines"
        self.files = [self.sel_files[i].split("/")[-1].split(".")[0] for i in range(len((self.sel_files)))]
        mcmc_params = {"nb_chain"   : self.mcmc_vars[0][1].get(),
                       "nb_iter"    : self.mcmc_vars[1][1].get(),
                       "nb_burn"    : self.mcmc_vars[2][1].get(),
                       "thin"       : self.mcmc_vars[3][1].get(),
                       "tune_inter" : self.mcmc_vars[4][1].get(),
                       "prop_scale" : self.mcmc_vars[5][1].get(),
                       "verbose"    : self.check_vars[4][1].get(),
                        }
        # Appel de la fonction d'inversion avec les paramètres sélectionnés
        try:    del(self.all_results)
        except: pass
        self.all_results = {}
        self.draw_drop_down()
#        self.var_review.set("Working...")
        for (i, self.f_n) in enumerate(self.files):
            print "====================="
            self.activity()
#            self.var_review.set(f_n)
            
            sol = mcmcSIPinv(   self.mod.get(), self.sel_files[i], mcmc = mcmc_params,
                                headers=self.head.get(), ph_units=self.units.get(),
                                cc_modes=self.modes_n.get(), decomp_poly=self.poly_n.get(),
                                c_exp=self.c_exp.get(), keep_traces=self.check_vars[3][1].get())

            self.all_results[self.f_n] = {"pm":sol["params"],"MDL":sol["pymc_model"],"data":sol["data"],"fit":sol["fit"], "sol":sol}
    #         Impression ou non des résultats, graphiques, histogrammes
#            update_results(sol)
#            iR.print_resul(sol)
#            iR.save_resul(sol)
#            fig_fit = iR.plot_fit(sol["data"], sol["fit"], model, f_n, save=check_vars[1][1].get(), draw=check_vars[0][1].get())
#            if check_vars[0][1].get():
#                plot_window(fig_fit, "Inversion results: "+f_n)
#            if model == "PDecomp":
#                iR.plot_debye(sol, save=check_vars[1][1].get(), draw=False)
#            if model == "DDebye":
#                iR.plot_debye_histo(sol, save=check_vars[1][1].get(), draw=False)
#            if check_vars[2][1].get():
#                iR.plot_histo(all_results[f_n]["sol"], save=True)
#                iR.plot_traces(all_results[f_n]["sol"], save=True)
#                iR.plot_summary(all_results[f_n]["sol"], save=True)
#                iR.plot_autocorr(all_results[f_n]["sol"], save=True)
            if self.files.index(self.f_n)+1 == len(self.files):
                self.activity(done=True)
#                diagn_buttons()
#                write_output_path()
#                print "====================="
#                if len(files) > 1:
#                    iR.merge_results(model,files)
#                    print "====================="


    def make_main_frames(self):
        
        self.frame_import = tk.LabelFrame(self.master, text="1. Import data", width=200, height=5, font=fontz["bold"])
        self.frame_import.grid(row = 0, column=1, columnspan=1, rowspan=5, sticky=tk.W+tk.E+tk.N, padx=10, pady=(5,15))     
        self.frame_import.grid_columnconfigure(0, weight=1), self.frame_import.grid_rowconfigure(0, weight=1)
        
        self.frame_model = tk.LabelFrame(self.master, text="2. SIP model", width=200, height=4, font=fontz["bold"])
        self.frame_model.grid(row = 6, column=1, columnspan=1, sticky=tk.W+tk.E+tk.N, padx=10, pady=(5,15))        
        self.frame_model.grid_columnconfigure(0, weight=1), self.frame_model.grid_rowconfigure(0, weight=1)
        
        self.frame_mcmc = tk.LabelFrame(self.master, text="3. MCMC settings", width=200, height=4, font=fontz["bold"])
        self.frame_mcmc.grid(row = 7, column=1, columnspan=1, sticky=tk.W+tk.E+tk.N, padx=10, pady=(5,15), ipady=3)        
        self.frame_mcmc.grid_columnconfigure(0, weight=1)

        self.frame_ruex = tk.LabelFrame(self.master, text="4. Options", width=200, height=4, font=fontz["bold"])
        self.frame_ruex.grid(row = 8, column=1, columnspan=1, sticky=tk.S+tk.W+tk.E+tk.N, padx=10, pady=(5,10))            
        self.frame_ruex.columnconfigure(0, weight=1)
            
            
        self.frame_list = tk.LabelFrame(self.master, text="List of imported files", font=fontz["bold"])
        self.frame_list.grid(row = 0, column=0, columnspan=1, rowspan=15, sticky=tk.W+tk.E+tk.N+tk.S, padx=10, pady=(5,10))
        self.frame_list.grid_rowconfigure(1, weight=1), self.frame_list.columnconfigure(0, weight=1)

        self.frame_results = tk.LabelFrame(self.master, text="Results", font=fontz["bold"])
        self.frame_results.grid(row = 0, column=2, columnspan=1, rowspan=15, sticky=tk.W+tk.E+tk.N+tk.S, padx=10, pady=(5,10))
        self.frame_results.grid_rowconfigure(14, weight=1)
        
    def make_browse_button(self):
        self.but_browse = tk.Button(self.frame_import, height=1, text = "Open file browser", bg='gray97',
                            command = self.get_file_list, relief=tk.GROOVE)
        self.but_browse.grid(row=0, column=0, columnspan=2, pady=(10,10), padx=(10,10), sticky=tk.E+tk.W)

    def headers_input(self):
        self.head = tk.IntVar()
        self.head.set(self.root_ini["Nb header lines"])  # set last used value
        tk.Label(self.frame_import, text="""Nb header lines:""", justify = tk.LEFT).grid(row=1, column=0, columnspan=1,sticky=tk.W, padx=(10,0))
        tk.Entry(self.frame_import, textvariable=self.head, width=12).grid(row=1, column=1, columnspan=1,sticky=tk.E+tk.W,padx=(10,10))
        
    def make_options(self):
        self.check_vars = [("Draw data and fit", tk.BooleanVar()),
                           ("Save fit figure"  , tk.BooleanVar()),
                          ("Save QC figures " , tk.BooleanVar()),
                          ("Keep txt traces"  , tk.BooleanVar()),
                          ("Tuning verbose"   , tk.BooleanVar()),]
        self.frame_checkbox = tk.Frame(self.frame_ruex)
        self.frame_checkbox.grid(row=0, column=0, rowspan=2, pady=(5,10), padx=10)
        for i, (k, v) in enumerate(self.check_vars):
            tk.Checkbutton(self.frame_checkbox, text=k, variable=v).grid(row=i, column=0, sticky=tk.W+tk.N+tk.S)
            self.check_vars[i][1].set(self.root_ini[k]) # set last used values

    def set_plot_par(self): # Setting up plotting parameters
        try:
            print "\nLoading plot parameters"
            rcParams.update(iR.plot_par())
            print "Plot parameters successfully loaded"
        except:
            print "Plot parameters not found, using default values"
        stdout.flush()

    def mcmc_parameters(self):
        # McMC parameters
        self.mcmc_vars = [("Nb of chains:"    , tk.IntVar()),
                          ("Total iterations:", tk.IntVar()),
                          ("Burn-in period:"  , tk.IntVar()),
                          ("Thinning factor:" , tk.IntVar()),
                          ("Tuning interval:" , tk.IntVar()),
                          ("Proposal scale:"  , tk.DoubleVar()),]
        for i, (k, v) in enumerate(self.mcmc_vars):
            v.set(self.root_ini[k])
            tk.Label(self.frame_mcmc, text=k, justify = tk.LEFT).grid(row=i, column=0, sticky=tk.W, padx=(10,0))
            tk.Entry(self.frame_mcmc ,textvariable=v, width=12).grid(row=i, column=1, sticky=tk.E,padx=(0,10))

    def phase_units(self):        
        # Phase units
        unites = ["mrad", "rad" ,"deg"]
        self.units = tk.StringVar()
        self.units.set(self.root_ini["Phase units"])  # set last used value
        tk.Label(self.frame_import, text="""Phase units:""", justify = tk.LEFT).grid(row=2, column=0, sticky=tk.W, padx=(10,0))
        for i, u in enumerate(unites):
            tk.Radiobutton(self.frame_import, text=u, variable=self.units, command=None, value=u).grid(row=i+2, column=1, sticky=tk.W, padx=(10,10), pady=pad_radio)

    def draw_rtd_check(self):
        try:    self.mod_opt_frame.destroy()
        except: pass
        self.mod_opt_frame = tk.Frame(self.frame_model)
        self.mod_opt_frame.grid(row=0, column=1, rowspan=4)
#        mod_opt_frame.grid_rowconfigure(4, weight=1)
        if self.mod.get() == "PDecomp":
            poly_lab = tk.Label(self.mod_opt_frame, text="""Polyn order""", justify = tk.LEFT, font=fontz["normal_small"])
            poly_lab.grid(row=0, column=1, rowspan=1, sticky=tk.W+tk.N, pady=(0,0), padx=(0,10))
            poly_scale = tk.Scale(self.mod_opt_frame, variable=self.poly_n, width=10, length=70, from_=3, to=5, font=fontz["normal_small"], orient=tk.HORIZONTAL)
            poly_scale.grid(row=1, column=1, rowspan=1, sticky=tk.E+tk.N, padx=(0,10), pady=(0,0))
            exp_lab = tk.Label(self.mod_opt_frame, text="""c exponent""", justify = tk.LEFT, font=fontz["normal_small"])
            exp_lab.grid(row=2, column=1, rowspan=1, sticky=tk.W+tk.N, pady=(0,0), padx=(0,10))
            exp_scale = tk.Scale(self.mod_opt_frame, variable=self.c_exp, width=10, length=70, from_=0.1, to=1.0, resolution=0.05, font=fontz["normal_small"], orient=tk.HORIZONTAL)
            exp_scale.grid(row=3, column=1, rowspan=1, sticky=tk.E+tk.N, padx=(0,10), pady=(0,0))
        if self.mod.get() == "ColeCole":
            modes_lab = tk.Label(self.mod_opt_frame, text="""Nb modes""", justify = tk.LEFT, font=fontz["normal_small"])
            modes_lab.grid(row=0, column=1, rowspan=1, sticky=tk.W+tk.N, pady=(0,0), padx=(0,10))
            modes_scale = tk.Scale(self.mod_opt_frame, variable=self.modes_n, width=10, length=70, from_=1, to=3, font=fontz["normal_small"], orient=tk.HORIZONTAL)
            modes_scale.grid(row=1, column=1, rowspan=1, sticky=tk.E+tk.N, padx=(0,10), pady=(0,0))



    def model_choice(self):
        # Available models
        models = [("Pelton \nCole-Cole","ColeCole"),
                  ("Dias \nmodel","Dias"),
                  ("Debye / Warburg \ndecomposition","PDecomp"),]
        self.modes_n, self.poly_n, self.c_exp = tk.IntVar(), tk.IntVar(), tk.DoubleVar()
        self.modes_n.set(self.root_ini["Nb modes"]), self.poly_n.set(self.root_ini["Polyn order"]), self.c_exp.set(self.root_ini["Freq dep"])

        ### Model choice
        self.mod = tk.StringVar()
        self.mod.set(self.root_ini["Spectral IP model"])  # set last used values
        for i, (txt, val) in enumerate(models):
            tk.Radiobutton(self.frame_model, text=txt, justify=tk.LEFT, variable = self.mod, command = self.draw_rtd_check, value=val).grid(row=i, column=0, sticky=tk.W+tk.S, padx=(10,0), pady=pad_radio+2)
        self.draw_rtd_check()


    def run_exit(self): # self expl
        tk.Button(self.frame_ruex, width=14, text = "RUN", fg='blue', bg='gray97', font=fontz["bold"], relief=tk.GROOVE,
                            command = self.run_inversion).grid(row=0, column=1, sticky=tk.N+tk.E+tk.S, padx=(0,10), pady=(5,0))
    
        tk.Button(self.frame_ruex, width=14, text = "EXIT", fg='red', bg='gray97', font=fontz["bold"], relief=tk.GROOVE,
                            command = self.master.destroy).grid(row=1, column=1, sticky=tk.S+tk.E+tk.N, padx=(0,10), pady=(0,10))

    def plot_window(self, fig, name=""):
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

    def preview_data(self):
        try:
            for i in self.text_files.curselection():
                sel = str(self.open_files[i])
                fn = sel.split("/")[-1].split(".")[0]
                fig_data = iR.plot_data(sel, self.head.get(), self.uni.get())
                self.plot_window(fig_data, "Data preview: "+fn)
        except:
            tkMessageBox.showwarning("Preview error",
                                     "Can't draw data\nImport and select at least one data file first", parent=self.master)

    def get_file_list(self):
        files = tkFileDialog.askopenfilenames(parent=self.master,title='Choose a file')
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







    working_path = str(osp_dirname(osp_realpath(argv[0]))).replace("\\", "/")+"/"
    default_root = {
                    "Spectral IP model" : "ColeCole",
                    "Nb of chains:"     : 1,
                    "Total iterations:" : 10000,
                    "Burn-in period:"   : 8000,
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
                    "Imported files"    : [],
                    "Polyn order"       : 4,
                    "Freq dep"          : 1.0,
                    "Nb modes"          : 2,
                    }
    
    def load(self):
        print "\nLoading root_ini from:\n", self.working_path
        try:
            with open(self.working_path+'root_ini') as f:
                self.root_ini = jload(f)
            print "root_ini successfully loaded"
        except:
            print "root_ini not found, using default values"
            self.root_ini = self.default_root
        stdout.flush()
        
    def save(self):
        root_save = {
                        "Spectral IP model" : self.mod.get(),
                        "Nb of chains:"     : self.mcmc_vars[0][1].get(),
                        "Total iterations:" : self.mcmc_vars[1][1].get(),
                        "Burn-in period:"   : self.mcmc_vars[2][1].get(),
                        "Thinning factor:"  : self.mcmc_vars[3][1].get(),
                        "Tuning interval:"  : self.mcmc_vars[4][1].get(),
                        "Proposal scale:"   : self.mcmc_vars[5][1].get(),
                        "Nb header lines"   : self.head.get(),
                        "Phase units"       : self.units.get(),
                        "Keep txt traces"   : self.check_vars[3][1].get(),
                        "Draw data and fit" : self.check_vars[0][1].get(),
                        "Save QC figures "  : self.check_vars[2][1].get(),
                        "Save fit figure"   : self.check_vars[1][1].get(),
                        "Tuning verbose"    : self.check_vars[4][1].get(),
                        "Imported files"    : self.open_files,
                        "Polyn order"       : self.poly_n.get(),
                        "Freq dep"          : self.c_exp.get(),
                        "Nb modes"          : self.modes_n.get(),
                    }
        print "\nSaving root_ini"
        with open(self.working_path+'root_ini', 'w') as f:
            jdump(root_save, f)
        print "root_ini successfully saved in:\n", self.working_path
        print "\n=========END========="

  
class Demo2:
    def __init__(self, master):
        self.master = master
        self.frame = tk.Frame(self.master)
        self.quitButton = tk.Button(self.frame, text = 'Quit', width = 25, command = self.close_windows)
        self.quitButton.pack()
        self.frame.pack()
    def close_windows(self):
        self.master.destroy()

def main(): 
    root = tk.Tk()
    app = MainApplication(root, fontz)
    #==============================================================================
    # Window size options
    root.mainloop()
    app.save()
if __name__ == '__main__':
    main()