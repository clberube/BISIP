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
        
        self.load()     
        
        self.frame_import = tk.LabelFrame(self.master, text="1. Import data", width=200, height=5, font=fontz["bold"])
        self.frame_import.grid(row = 0, column=1, columnspan=1, rowspan=5, sticky=tk.W+tk.E+tk.N, padx=10, pady=(5,15))     
        
        self.frame_model = tk.LabelFrame(self.master, text="2. SIP model", width=200, height=4, font=fontz["bold"])
        self.frame_model.grid(row = 6, column=1, columnspan=1, sticky=tk.W+tk.E+tk.N, padx=10, pady=(5,15))        
        
        self.frame_mcmc = tk.LabelFrame(self.master, text="3. MCMC settings", width=200, height=4, font=fontz["bold"])
        self.frame_mcmc.grid(row = 7, column=1, columnspan=1, sticky=tk.W+tk.E+tk.N, padx=10, pady=(5,15), ipady=3)        

        self.frame_ruex = tk.LabelFrame(self.master, text="4. Options", width=200, height=4, font=fontz["bold"])
        self.frame_ruex.grid(row = 8, column=1, columnspan=1, sticky=tk.S+tk.W+tk.E+tk.N, padx=10, pady=(5,10))            
            
        self.frame_list = tk.LabelFrame(self.master, text="List of imported files", font=fontz["bold"])
        self.frame_list.grid(row = 0, column=0, columnspan=1, rowspan=15, sticky=tk.W+tk.E+tk.N+tk.S, padx=10, pady=(5,10))
        
        self.frame_results = tk.LabelFrame(self.master, text="Results", font=fontz["bold"])
        self.frame_results.grid(row = 0, column=2, columnspan=1, rowspan=15, sticky=tk.W+tk.E+tk.N+tk.S, padx=10, pady=(5,10))
        

        self.but_browse = tk.Button(self.frame_import, height=1, text = "Open file browser", bg='gray97',
                            command = self.get_file_list, relief=tk.GROOVE)
        self.but_browse.grid(row=0, column=0, columnspan=2, pady=(10,10), padx=(10,10), sticky=tk.E+tk.W)

        self.head = tk.IntVar()
        self.head.set(self.root_ini["Nb header lines"])  # set last used value
        tk.Label(self.frame_import, text="""Nb header lines:""", justify = tk.LEFT).grid(row=1, column=0, columnspan=1,sticky=tk.W, padx=(10,0))
        tk.Entry(self.frame_import, textvariable=self.head, width=12).grid(row=1, column=1, columnspan=1,sticky=tk.E+tk.W,padx=(10,10))
        
        # Phase units
        unites = ["mrad", "rad" ,"deg"]
        self.uni = tk.StringVar()
        self.uni.set(self.root_ini["Phase units"])  # set last used value
        tk.Label(self.frame_import, text="""Phase units:""", justify = tk.LEFT).grid(row=2, column=0, sticky=tk.W, padx=(10,0))
        for i, u in enumerate(unites):
            tk.Radiobutton(self.frame_import, text=u, variable=self.uni, command=None, value=u).grid(row=i+2, column=1, sticky=tk.W, padx=(10,10), pady=pad_radio)


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

        self.open_files = self.root_ini["Imported files"]
        self.draw_file_list()
        self.run_exit()
        

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

    def run_exit(self): # self expl
        tk.Button(self.frame_ruex, width=14, text = "RUN", fg='blue', bg='gray97', font=fontz["bold"], relief=tk.GROOVE,
                            command = None).grid(row=0, column=1, sticky=tk.N+tk.E+tk.S, padx=(0,10), pady=(5,0))
    
        tk.Button(self.frame_ruex, width=14, text = "EXIT", fg='red', bg='gray97', font=fontz["bold"], relief=tk.GROOVE,
                            command = self.master.destroy).grid(row=1, column=1, sticky=tk.S+tk.E+tk.N, padx=(0,10), pady=(0,10))

    def get_file_list(self):
        files = tkFileDialog.askopenfilenames(parent=self.master,title='Choose a file')
        self.open_files = sorted(list(files))
        self.draw_file_list()

    def draw_file_list(self):

        frame_path = tk.Frame(self.frame_list)
        frame_path.grid(row=0, column=0, padx=20, pady=10, sticky=tk.E+tk.W)
        frame_path.columnconfigure(2, weight=1)
    
        frame_select = tk.Frame(self.frame_list)
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

        # Fill list box
        if len(self.open_files) >= 1: # Only draw this listbox if files were selected in browser
            for (i, o) in enumerate(self.open_files):
                fil = o.split("/")[-1] # Split full path to keep only name and extension
                text_files.insert(i, fil) # Insert items 1 by 1 in loop
            locs = "/".join(self.open_files[0].split("/")[:-1])+"/" # Path to imported files in string format
            text_loc.insert("1.0", locs) # Insert path to empty text box
            text_files.select_set(0) # Select first item in list after populating
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
                        "Spectral IP model" : mod.get(),
                        "Nb of chains:"     : self.mcmc_vars[0][1].get(),
                        "Total iterations:" : self.mcmc_vars[1][1].get(),
                        "Burn-in period:"   : self.mcmc_vars[2][1].get(),
                        "Thinning factor:"  : self.mcmc_vars[3][1].get(),
                        "Tuning interval:"  : self.mcmc_vars[4][1].get(),
                        "Proposal scale:"   : self.mcmc_vars[5][1].get(),
                        "Nb header lines"   : self.head.get(),
                        "Phase units"       : self.uni.get(),
                        "Keep txt traces"   : self.check_vars[3][1].get(),
                        "Draw data and fit" : self.check_vars[0][1].get(),
                        "Save QC figures "  : self.check_vars[2][1].get(),
                        "Save fit figure"   : self.check_vars[1][1].get(),
                        "Tuning verbose"    : self.check_vars[4][1].get(),
                        "Imported files"    : self.open_files,
                        "Polyn order"       : poly_n.get(),
                        "Freq dep"          : c_exp.get(),
                        "Nb modes"          : modes_n.get(),
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
    root.mainloop()
    app.save()
if __name__ == '__main__':
    main()